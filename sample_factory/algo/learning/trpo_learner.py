from __future__ import annotations

import glob
import os
import time
from abc import ABC, abstractmethod
from os.path import join
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import gae_advantages, prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.cfg.configurable import Configurable
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.decay import LinearDecay
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir, log


class LearningRateScheduler:
    def update(self, current_lr, recent_kls):
        return current_lr

    def invoke_after_each_minibatch(self):
        return False

    def invoke_after_each_epoch(self):
        return False


class KlAdaptiveScheduler(LearningRateScheduler, ABC):
    def __init__(self, cfg: Config):
        self.lr_schedule_kl_threshold = cfg.lr_schedule_kl_threshold
        self.min_lr = cfg.lr_adaptive_min
        self.max_lr = cfg.lr_adaptive_max

    @abstractmethod
    def num_recent_kls_to_use(self) -> int:
        pass

    def update(self, current_lr, recent_kls):
        num_kls_to_use = self.num_recent_kls_to_use()
        kls = recent_kls[-num_kls_to_use:]
        mean_kl = np.mean(kls)
        lr = current_lr
        if mean_kl > 2.0 * self.lr_schedule_kl_threshold:
            lr = max(current_lr / 1.5, self.min_lr)
        if mean_kl < (0.5 * self.lr_schedule_kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class KlAdaptiveSchedulerPerMinibatch(KlAdaptiveScheduler):
    def num_recent_kls_to_use(self) -> int:
        return 1

    def invoke_after_each_minibatch(self):
        return True


class KlAdaptiveSchedulerPerEpoch(KlAdaptiveScheduler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_minibatches_per_epoch = cfg.num_batches_per_epoch

    def num_recent_kls_to_use(self) -> int:
        return self.num_minibatches_per_epoch

    def invoke_after_each_epoch(self):
        return True


class LinearDecayScheduler(LearningRateScheduler):
    def __init__(self, cfg):
        num_updates = cfg.train_for_env_steps // cfg.batch_size * cfg.num_epochs
        self.linear_decay = LinearDecay([(0, cfg.learning_rate), (num_updates, 0)])
        self.step = 0

    def invoke_after_each_minibatch(self):
        return True

    def update(self, current_lr, recent_kls):
        self.step += 1
        lr = self.linear_decay.at(self.step)
        return lr


def get_lr_scheduler(cfg) -> LearningRateScheduler:
    if cfg.lr_schedule == "constant":
        return LearningRateScheduler()
    elif cfg.lr_schedule == "kl_adaptive_minibatch":
        return KlAdaptiveSchedulerPerMinibatch(cfg)
    elif cfg.lr_schedule == "kl_adaptive_epoch":
        return KlAdaptiveSchedulerPerEpoch(cfg)
    elif cfg.lr_schedule == "linear_decay":
        return LinearDecayScheduler(cfg)
    else:
        raise RuntimeError(f"Unknown scheduler {cfg.lr_schedule}")


def model_initialization_data(
        cfg: Config, policy_id: PolicyID, actor_critic: Module, policy_version: int, device: torch.device
) -> InitModelData:
    # in serial mode we will just use the same actor_critic directly
    state_dict = None if cfg.serial_mode else actor_critic.state_dict()
    model_state = (policy_id, state_dict, device, policy_version)
    return model_state


class TRPOLearner(Configurable):
    def __init__(
            self,
            cfg: Config,
            env_info: EnvInfo,
            policy_versions_tensor: Tensor,
            policy_id: PolicyID,
            param_server: ParameterServer,
    ):
        Configurable.__init__(self, cfg)

        self.timing = Timing(name=f"Learner {policy_id} profile")

        self.policy_id = policy_id

        self.env_info = env_info

        self.safety_bound = env_info.safety_bound / env_info.timeout * env_info.frameskip

        self.device = None
        self.actor_critic: Optional[ActorCritic] = None

        self.optimizer = None

        self.curr_lr: Optional[float] = None
        self.lr_scheduler: Optional[LearningRateScheduler] = None

        self.train_step: int = 0  # total number of updates
        self.env_steps: int = 0  # total number of environment steps consumed by the learner

        self.best_performance = -1e9

        # for configuration updates, i.e. from PBT
        self.new_cfg: Optional[Dict] = None

        # for multi-policy learning (i.e. with PBT) when we need to load weights of another policy
        self.policy_to_load: Optional[PolicyID] = None

        # decay rate at which summaries are collected
        # save summaries every 5 seconds in the beginning, but decay to every 4 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self.summary_rate_decay_seconds = LinearDecay([(0, 2), (100000, 60), (1000000, 120)])
        self.last_summary_time = 0
        self.last_milestone_time = 0

        # shared tensor used to share the latest policy version between processes
        self.policy_versions_tensor: Tensor = policy_versions_tensor

        self.param_server: ParameterServer = param_server

        self.is_initialized = False

    def init(self) -> InitModelData:
        if self.cfg.exploration_loss_coeff == 0.0:
            self.exploration_loss_func = lambda action_distr, valids, num_invalids: 0.0
        elif self.cfg.exploration_loss == "entropy":
            self.exploration_loss_func = self._entropy_exploration_loss
        elif self.cfg.exploration_loss == "symmetric_kl":
            self.exploration_loss_func = self._symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f"{self.cfg.exploration_loss} not supported!")

        if self.cfg.seed is not None:
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        self.device = policy_device(self.cfg, self.policy_id)

        log.debug("Initializing actor-critic model on device %s", self.device)

        # trainable torch module
        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self.actor_critic)
        self.actor_critic.model_to_device(self.device)

        def share_mem(t):
            if t is not None and not t.is_cuda:
                return t.share_memory_()
            return t

        # noinspection PyProtectedMember
        self.actor_critic._apply(share_mem)
        self.actor_critic.train()

        params = list(self.actor_critic.critic.parameters())

        self.optimizer = self.create_optimizer(params)

        self.param_server.init(self.actor_critic, self.train_step, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        self.lr_scheduler = get_lr_scheduler(self.cfg)
        self.curr_lr = self.cfg.learning_rate if self.curr_lr is None else self.curr_lr
        self._apply_lr(self.curr_lr)

        self.is_initialized = True

        return model_initialization_data(self.cfg, self.policy_id, self.actor_critic, self.train_step, self.device)

    def create_optimizer(self, params):
        optimizer_cls = dict(adam=torch.optim.Adam)
        if self.cfg.optimizer not in optimizer_cls:
            raise RuntimeError(f"Unknown optimizer {self.cfg.optimizer}")
        optimizer_cls = optimizer_cls[self.cfg.optimizer]
        log.debug(f"Using optimizer {optimizer_cls} for value function")
        optimizer_kwargs = dict(
            lr=self.cfg.learning_rate,  # Use default lr only in ctor, then we use the one loaded from the checkpoint
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )
        if self.cfg.learning_rate in ["adam", "lamb"]:
            optimizer_kwargs["eps"] = self.cfg.adam_eps
        return optimizer_cls(params, **optimizer_kwargs)

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f"checkpoint_p{policy_id}")
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern="checkpoint_*"):
        checkpoints = glob.glob(join(checkpoints_dir, pattern))
        return sorted(checkpoints)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning("No checkpoints found")
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                # noinspection PyBroadException
                try:
                    log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f"Could not load from checkpoint, attempt {attempt}")

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict["train_step"]
            self.env_steps = checkpoint_dict["env_steps"]
            self.best_performance = checkpoint_dict.get("best_performance", self.best_performance)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.curr_lr = checkpoint_dict.get("curr_lr", self.cfg.learning_rate)

        log.info(f"Loaded experiment state at {self.train_step=}, {self.env_steps=}")

    def load_from_checkpoint(self, policy_id: PolicyID, load_progress: bool = True) -> None:
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern=f"{name_prefix}_*")
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug("Did not load from checkpoint, starting from scratch!")
        else:
            log.debug("Loading model from checkpoint")

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def _should_save_summaries(self):
        summaries_every_seconds = self.summary_rate_decay_seconds.at(self.train_step)
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1

    def _get_checkpoint_dict(self):
        checkpoint = {
            "train_step": self.train_step,
            "env_steps": self.env_steps,
            "best_performance": self.best_performance,
            "model": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "curr_lr": self.curr_lr,
        }
        return checkpoint

    def _save_impl(self, name_prefix, name_suffix, keep_checkpoints, verbose=True) -> bool:
        if not self.is_initialized:
            return False

        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, f"{name_prefix}_temp")
        checkpoint_name = f"{name_prefix}_{self.train_step:09d}_{self.env_steps}{name_suffix}.pth"
        filepath = join(checkpoint_dir, checkpoint_name)
        if verbose:
            log.info("Saving %s...", filepath)

        # This should protect us from a rare case where something goes wrong mid-save and we end up with a corrupted
        # checkpoint file. It better be a corrupted temp file.
        torch.save(checkpoint, tmp_filepath)
        os.rename(tmp_filepath, filepath)

        while len(checkpoints := self.get_checkpoints(checkpoint_dir, f"{name_prefix}_*")) > keep_checkpoints:
            oldest_checkpoint = checkpoints[0]
            if os.path.isfile(oldest_checkpoint):
                if verbose:
                    log.debug("Removing %s", oldest_checkpoint)
                os.remove(oldest_checkpoint)

        return True

    def save(self) -> bool:
        return self._save_impl("checkpoint", "", self.cfg.keep_checkpoints)

    def save_milestone(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None
        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        checkpoint_name = f"checkpoint_{self.train_step:09d}_{self.env_steps}.pth"

        milestones_dir = ensure_dir_exists(join(checkpoint_dir, "milestones"))
        milestone_path = join(milestones_dir, f"{checkpoint_name}")
        log.info("Saving a milestone %s", milestone_path)
        torch.save(checkpoint, milestone_path)

    def save_best(self, policy_id, metric, metric_value) -> bool:
        if policy_id != self.policy_id:
            return False
        p = 3  # precision, number of significant digits
        if metric_value - self.best_performance > 1 / 10 ** p:
            log.info(f"Saving new best policy, {metric}={metric_value:.{p}f}!")
            self.best_performance = metric_value
            name_suffix = f"_{metric}_{metric_value:.{p}f}"
            return self._save_impl("best", name_suffix, 1, verbose=False)

        return False

    def set_new_cfg(self, new_cfg: Dict) -> None:
        self.new_cfg = new_cfg

    def set_policy_to_load(self, policy_to_load: PolicyID) -> None:
        self.policy_to_load = policy_to_load

    def _maybe_update_cfg(self) -> None:
        if self.new_cfg is not None:
            for key, value in self.new_cfg.items():
                if getattr(self.cfg, key) != value:
                    log.debug("Learner %d replacing cfg parameter %r with new value %r", self.policy_id, key, value)
                    setattr(self.cfg, key, value)

            if self.cfg.lr_schedule == "constant" and self.curr_lr != self.cfg.learning_rate:
                # PBT-optimized learning rate, only makes sense if we use constant LR
                # in case of more advanced LR scheduling we should update the parameters of the scheduler, not the
                # learning rate directly
                log.debug(f"Updating learning rate from {self.curr_lr} to {self.cfg.learning_rate}")
                self.curr_lr = self.cfg.learning_rate
                self._apply_lr(self.curr_lr)

            for param_group in self.optimizer.param_groups:
                param_group["betas"] = (self.cfg.adam_beta1, self.cfg.adam_beta2)
                log.debug("Optimizer lr value %.7f, betas: %r", param_group["lr"], param_group["betas"])

            self.new_cfg = None

    def _maybe_load_policy(self) -> None:
        cfg = self.cfg
        if self.policy_to_load is not None:
            with self.param_server.policy_lock:
                # don't re-load progress if we are loading from another policy checkpoint
                self.load_from_checkpoint(self.policy_to_load, load_progress=False)

            # make sure everything (such as policy weights) is committed to shared device memory
            synchronize(cfg, self.device)
            # this will force policy update on the inference worker (policy worker)
            # we add max_policy_lag steps so that all experience currently in batches is invalidated
            self.train_step += cfg.max_policy_lag + 1
            self.policy_versions_tensor[self.policy_id] = self.train_step

            self.policy_to_load = None

        timestamp = cfg.load_checkpoint_timestamp
        if timestamp:
            level = cfg.load_checkpoint_level
            kind = cfg.load_checkpoint_kind
            name_prefix = dict(latest="checkpoint", best="best")[kind]
            checkpoints_dir = f'{cfg.train_dir}/{cfg.algo}/{cfg.env}/Level_{level}/{timestamp}/checkpoint_p0'
            print(f'Loading checkpoint from {checkpoints_dir}')
            if not os.path.exists(checkpoints_dir):
                raise FileNotFoundError(f"No checkpoint directory found at {checkpoints_dir}")
            checkpoints = TRPOLearner.get_checkpoints(checkpoints_dir, f"{name_prefix}_*")
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint files match the specified pattern.")
            checkpoint_dict = TRPOLearner.load_checkpoint(checkpoints, self.device)
            self.actor_critic.load_state_dict(checkpoint_dict["model"])
            cfg.load_checkpoint_timestamp = None  # Set to None to prevent from loading a second time

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            buff, experience_size, num_invalids = self._prepare_batch(batch)

        if num_invalids >= experience_size:
            log.error(f"Learner {self.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self.timing.add_time("train"):
                train_stats = self._train(buff, self.cfg.batch_size, experience_size, num_invalids)

            # multiply the number of samples by frameskip so that FPS metrics reflect the number
            # of environment steps actually simulated
            if self.cfg.summaries_use_frameskip:
                self.env_steps += experience_size * self.env_info.frameskip
            else:
                self.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
            if train_stats is not None:
                if train_stats is not None:
                    stats[TRAIN_STATS] = train_stats
                stats[STATS_KEY] = memory_stats("learner", self.device)

            return stats

    def _train(self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int) -> Optional[
        Dict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_loss = 1e9
            epoch_losses = [0] * self.cfg.num_batches_per_epoch

            num_optimization_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]
                    mb = self._get_minibatch(gpu_buffer, indices)
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        surrogate_loss,
                        exploration_loss,
                        value_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    actor_loss = surrogate_loss + exploration_loss
                    total_loss = actor_loss + value_loss
                    epoch_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(total_loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(total_loss),
                            to_scalar(surrogate_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with timing.add_time("update"):
                    self._trpo_step(mb, actor_loss, value_loss, num_invalids, mb.valids)
                    num_optimization_steps += 1

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    # Update the learning rate scheduler if necessary
                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, None)

                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        del summary_vars
                        force_summaries = False

                    synchronize(self.cfg, self.device)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            new_epoch_loss = float(np.mean(epoch_losses))
            loss_delta_abs = abs(prev_epoch_loss - new_epoch_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d optimization steps), loss delta %.7f",
                    epoch + 1,
                    num_optimization_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_loss = new_epoch_loss

        return stats_and_summaries

    def _trpo_step(self, mb, actor_loss, value_loss, num_invalids, valids):
        # Implement TRPO update using conjugate gradient and line search
        policy_params = [p for p in self.actor_critic.actor.parameters() if p.requires_grad]

        # Compute policy gradients
        loss = actor_loss
        for p in policy_params:
            p.grad = None
        loss.backward(retain_graph=True)

        # Flatten gradients
        grads = torch.cat([p.grad.view(-1) for p in policy_params]).detach()

        # Fisher vector product function
        def Fvp(v):
            kl = self._compute_kl(mb, valids, num_invalids)
            kl = kl.mean()

            # First-order gradient with respect to policy parameters
            kl_grad = torch.autograd.grad(kl, policy_params, create_graph=True)
            flat_kl_grad = torch.cat([g.contiguous().view(-1) for g in kl_grad])

            # Compute the directional derivative (gradient-vector product)
            kl_grad_v = (flat_kl_grad * v).sum()

            # Second-order gradient (Hessian-vector product)
            kl_hessian = torch.autograd.grad(kl_grad_v, policy_params, retain_graph=True)
            flat_kl_hessian = torch.cat([g.contiguous().view(-1) for g in kl_hessian]).detach()

            return flat_kl_hessian + self.cfg.cg_damping * v

        # Compute step direction using Conjugate Gradient
        step_dir = self._conjugate_gradient(Fvp, grads)

        # Compute step size
        shs = 0.5 * (step_dir * Fvp(step_dir)).sum(0, keepdim=True)
        max_step = torch.sqrt(self.cfg.max_kl / shs)[0]
        full_step = -step_dir * max_step

        # Line search to enforce KL constraint
        prev_params = self._get_flat_params_from(policy_params)
        success, new_params = self._line_search(
            mb, prev_params, full_step, actor_loss, valids, num_invalids
        )

        if success:
            self._set_flat_params_to(policy_params, new_params)
            log.info("Line search succeeded. Updating parameters...")
        else:
            log.warning("Line search failed. No parameter update performed.")

        # Update value function using standard gradient descent
        value_params = [p for p in self.actor_critic.critic.parameters()]
        for p in value_params:
            p.grad = None
        value_loss.backward()
        if self.cfg.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(value_params, self.cfg.max_grad_norm)
        self.optimizer.step()

    def _compute_kl(self, mb, valids, num_invalids):
        action_distribution = self.actor_critic.action_distribution()
        old_action_distribution = get_action_distribution(
            self.actor_critic.action_space,
            mb.action_logits,
        )
        kl = old_action_distribution.kl_divergence(action_distribution)
        kl = masked_select(kl, valids, num_invalids)
        return kl

    def _conjugate_gradient(self, Avp_func, b):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rdotr = r.dot(r)

        for _ in range(self.cfg.cg_nsteps):
            Avp = Avp_func(p)
            alpha = rdotr / (p.dot(Avp) + 1e-8)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = r.dot(r)
            if new_rdotr < self.cfg.cg_residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def _line_search(self, mb, prev_params, full_step, prev_loss, valids, num_invalids):
        stepfrac = 1.0
        accept_ratio = self.cfg.line_search_accept_ratio  # Not currently used
        params = [p for p in self.actor_critic.actor.parameters() if p.requires_grad]

        for _ in range(self.cfg.line_search_max_backtracks):
            new_params = prev_params + stepfrac * full_step
            self._set_flat_params_to(params, new_params)

            with torch.no_grad():
                # Recompute forward pass with updated parameters
                # Forward the observations through the network to get new action distributions
                actor_head_outputs, critic_head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
                if self.cfg.use_rnn:
                    # Rebuild RNN inputs if necessary
                    done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()

                    # Split rnn_states into actor and critic components
                    actor_rnn_states, critic_rnn_states = mb.rnn_states.chunk(2, dim=1)

                    # Build RNN inputs for actor
                    actor_head_output_seq, actor_rnn_states, actor_inverted_inds = build_rnn_inputs(
                        actor_head_outputs,
                        done_or_invalid,
                        actor_rnn_states,
                        self.cfg.recurrence,
                    )

                    # Build RNN inputs for critic
                    critic_head_output_seq, critic_rnn_states, critic_inverted_inds = build_rnn_inputs(
                        critic_head_outputs,
                        done_or_invalid,
                        critic_rnn_states,
                        self.cfg.recurrence,
                    )

                    with torch.backends.cudnn.flags(enabled=False):
                        actor_core_output_seq, critic_core_output_seq, _, _ = self.actor_critic.forward_core(
                            actor_head_output_seq, critic_head_output_seq, actor_rnn_states, critic_rnn_states)
                    actor_core_outputs = build_core_out_from_seq(actor_core_output_seq, actor_inverted_inds)
                    critic_core_outputs = build_core_out_from_seq(critic_core_output_seq, critic_inverted_inds)
                else:
                    actor_rnn_states, critic_rnn_states = mb.rnn_states[::self.cfg.recurrence].chunk(2, dim=1)
                    actor_core_outputs, critic_core_outputs, _, _ = self.actor_critic.forward_core(actor_head_outputs, critic_head_outputs,
                                                                                                   actor_rnn_states, critic_rnn_states)
                result = self.actor_critic.forward_tail(actor_core_outputs, critic_core_outputs, values_only=False,
                                                        sample_actions=False)
                action_distribution = self.actor_critic.action_distribution()
                log_prob_actions = action_distribution.log_prob(mb.actions)

                # Recompute surrogate loss
                adv = mb.advantages
                adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)
                surrogate_loss = -torch.mean(masked_select(log_prob_actions * adv, valids, num_invalids))

                # Recompute KL divergence
                kl_div = self._compute_kl(mb, valids, num_invalids).mean()

            # Check improvement and KL constraint
            loss_improve = prev_loss - surrogate_loss
            if loss_improve.item() > 0 and kl_div.item() <= self.cfg.max_kl:
                return True, new_params
            stepfrac *= self.cfg.line_search_backtrack_coeff

        # If line search fails, revert to previous parameters
        self._set_flat_params_to(params, prev_params)
        return False, prev_params

    def _get_flat_params_from(self, params):
        return torch.cat([p.data.view(-1) for p in params])

    def _set_flat_params_to(self, params, flat_params):
        prev_ind = 0
        for p in params:
            flat_size = int(np.prod(list(p.size())))
            p.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(p.size()))
            prev_ind += flat_size

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        var = train_loop_vars

        self.last_summary_time = time.time()
        stats = AttrDict()

        # Learning rates
        stats.lr = self.curr_lr
        stats.actual_lr = var.actual_lr  # Potentially scaled because of masked data

        # Update stats with actor-critic summaries
        stats.update(self.actor_critic.summaries())

        # Fraction of valid timesteps/actions
        stats.valids_fraction = var.mb.valids.float().mean()

        # Gradient norm
        grad_norm = (
                sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic.parameters() if
                    p.grad is not None) ** 0.5
        )
        stats.grad_norm = grad_norm

        # Loss components
        stats.loss = var.total_loss
        stats.value = var.values.mean()
        stats.entropy = var.action_distribution.entropy().mean()
        stats.policy_loss = var.surrogate_loss
        stats.kl_loss = var.kl_loss
        stats.value_loss = var.value_loss
        stats.exploration_loss = var.exploration_loss

        # Action statistics
        stats.act_min = var.mb.actions.min()
        stats.act_max = var.mb.actions.max()

        # Advantage statistics (if available)
        if "adv_mean" in stats:
            stats.adv_min = var.mb.advantages.min()
            stats.adv_max = var.mb.advantages.max()
            stats.adv_std = var.adv_std
            stats.adv_mean = var.adv_mean

        # Maximum absolute log probability of actions
        stats.max_abs_logprob = torch.abs(var.mb.action_logits).max()

        # Additional distribution summaries (if any)
        if hasattr(var.action_distribution, "summaries"):
            stats.update(var.action_distribution.summaries())

        # Epoch-end statistics for TRPO
        if var.epoch == self.cfg.num_epochs - 1 and var.batch_num == len(var.minibatches) - 1:
            # Compute value deltas between current and old value estimates
            value_delta = torch.abs(var.values - var.mb.values)
            value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

            # Value deltas
            stats.value_delta = value_delta_avg
            stats.value_delta_max = value_delta_max

        # Handling Adam optimizer's second moment to avoid numerical issues
        adam_max_second_moment = 0.0
        for key, tensor_state in self.optimizer.state.items():
            if "exp_avg_sq" in tensor_state:
                adam_max_second_moment = max(tensor_state["exp_avg_sq"].max().item(), adam_max_second_moment)
        stats.adam_max_second_moment = adam_max_second_moment

        # Policy version differences for versioning control
        version_diff = (var.curr_policy_version - var.mb.policy_version)[var.mb.policy_id == self.policy_id]
        stats.version_diff_avg = version_diff.mean()
        stats.version_diff_min = version_diff.min()
        stats.version_diff_max = version_diff.max()

        # Convert all stats to scalar values for logging
        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def _entropy_exploration_loss(self, action_distribution, valids, num_invalids: int) -> Tensor:
        entropy = action_distribution.entropy()
        entropy = masked_select(entropy, valids, num_invalids)
        entropy_loss = -self.cfg.exploration_loss_coeff * entropy.mean()
        return entropy_loss

    def _symmetric_kl_exploration_loss(self, action_distribution, valids, num_invalids: int) -> Tensor:
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = masked_select(kl_prior, valids, num_invalids).mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        kl_prior_loss = self.cfg.exploration_loss_coeff * kl_prior
        return kl_prior_loss

    def _optimizer_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def _apply_lr(self, lr: float) -> None:
        """Change learning rate in the optimizer."""
        if lr != self._optimizer_lr():
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def _get_minibatches(self, batch_size, experience_size):
        """Generating minibatches for training."""
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % batch_size == 0, f"experience size: {experience_size}, batch size: {batch_size}"
        minibatches_per_epoch = self.cfg.num_batches_per_epoch

        if minibatches_per_epoch == 1:
            return [None]  # single minibatch is actually the entire buffer, we don't need indices

        if self.cfg.shuffle_minibatches:
            # indices that will start the mini-trajectories from the same episode (for bptt)
            indices = np.arange(0, experience_size, self.cfg.recurrence)
            indices = np.random.permutation(indices)

            # complete indices of mini trajectories, e.g. with recurrence==4: [4, 16] -> [4, 5, 6, 7, 16, 17, 18, 19]
            indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
            indices = np.concatenate(indices)

            assert len(indices) == experience_size

            num_minibatches = experience_size // batch_size
            minibatches = np.split(indices, num_minibatches)
        else:
            minibatches = list(slice(i * batch_size, (i + 1) * batch_size) for i in range(0, minibatches_per_epoch))

            # this makes sense but I'd like to do some testing before enabling it
            # random.shuffle(minibatches)  # same minibatches between epochs, but in random order

        return minibatches

    @staticmethod
    def _get_minibatch(buffer, indices):
        if indices is None:
            # handle the case of a single batch, where the entire buffer is a minibatch
            return buffer

        mb = buffer[indices]
        return mb

    def _calculate_losses(
            self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Tensor, Dict]:
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            valids = mb.valids

        # Calculate policy head outside of recurrent loop
        with self.timing.add_time("forward_head"):
            actor_head_outputs, critic_head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = actor_head_outputs.size(0)

        # Initial RNN states
        with self.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                # Stop RNNs from backpropagating through invalid timesteps
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()

                # Split rnn_states into actor and critic components
                actor_rnn_states, critic_rnn_states = mb.rnn_states.chunk(2, dim=1)

                # Build RNN inputs for actor
                actor_head_output_seq, actor_rnn_states, actor_inverted_inds = build_rnn_inputs(
                    actor_head_outputs,
                    done_or_invalid,
                    actor_rnn_states,
                    self.cfg.recurrence,
                )

                # Build RNN inputs for critic
                critic_head_output_seq, critic_rnn_states, critic_inverted_inds = build_rnn_inputs(
                    critic_head_outputs,
                    done_or_invalid,
                    critic_rnn_states,
                    self.cfg.recurrence,
                )
            else:
                actor_rnn_states, critic_rnn_states = mb.rnn_states[::recurrence].chunk(2, dim=1)

        # Calculate RNN outputs for each timestep in a loop
        with self.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                with self.timing.add_time("bptt_forward_core"):
                    # Disable CuDNN during the RNN forward pass
                    with torch.backends.cudnn.flags(enabled=False):
                        actor_core_output_seq, critic_core_output_seq, _, _ = self.actor_critic.forward_core(
                            actor_head_output_seq, critic_head_output_seq, actor_rnn_states, critic_rnn_states)
                actor_core_outputs = build_core_out_from_seq(actor_core_output_seq, actor_inverted_inds)
                critic_core_outputs = build_core_out_from_seq(critic_core_output_seq, critic_inverted_inds)
                del actor_core_output_seq
                del critic_core_output_seq
            else:
                actor_core_outputs, critic_core_outputs, _, _ = self.actor_critic.forward_core(actor_head_outputs, critic_head_outputs, actor_rnn_states, critic_rnn_states)

            del actor_head_outputs
            del critic_head_outputs

        assert actor_core_outputs.shape[0] == minibatch_size

        with self.timing.add_time("tail"):
            # Calculate policy tail outside of recurrent loop
            result = self.actor_critic.forward_tail(actor_core_outputs, critic_core_outputs, values_only=False,
                                                    sample_actions=False)
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)

            # For TRPO, compute the old action distribution for KL divergence
            with torch.no_grad():
                old_action_distribution = get_action_distribution(self.actor_critic.action_space, mb.action_logits)

            values = result["values"].squeeze()

            del actor_core_outputs
            del critic_core_outputs

        # These computations are not part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            # Using regular GAE
            adv = mb.advantages
            targets = mb.returns
            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # Normalize advantage

        with self.timing.add_time("losses"):
            # Policy loss for TRPO
            masked_adv = masked_select(adv, valids, num_invalids)
            masked_log_prob_actions = masked_select(log_prob_actions, valids, num_invalids)
            policy_loss = -torch.mean(masked_adv * masked_log_prob_actions)

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)

            # KL divergence between new and old policies
            kl_divergence = old_action_distribution.kl_divergence(action_distribution)
            masked_kl_divergence = masked_select(kl_divergence, valids, num_invalids)
            kl_loss = torch.mean(masked_kl_divergence)

            # Value loss (mean squared error)
            masked_values = masked_select(values, valids, num_invalids)
            masked_targets = masked_select(targets, valids, num_invalids)
            value_loss = torch.nn.functional.mse_loss(masked_values, masked_targets)

        mean_cost = mb["costs"].mean()

        loss_summaries = dict(
            values=result["values"],
            avg_cost=mean_cost,
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            kl_loss=kl_loss,
        )

        return action_distribution, policy_loss, exploration_loss, value_loss, loss_summaries

    def _get_flat_params(self):
        params = []
        for param in self.actor_critic.parameters():
            params.append(param.data.view(-1))
        flat_params = torch.cat(params)
        return flat_params

    def _set_flat_params(self, new_params: torch.Tensor):
        prev_ind = 0
        for param in self.actor_critic.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(new_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

    def _get_flat_gradients(self) -> torch.Tensor:
        grads = []
        for param in self.actor_critic.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        return torch.cat(grads)

    def conjugate_gradient(self, fisher_vector_product_func, b, nsteps, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = fisher_vector_product_func(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def _compute_surrogate_loss(self, obs, actions, advantages, old_log_probs):
        outputs = self.actor_critic(obs)
        action_distribution = self.actor_critic.action_distribution()
        log_probs = action_distribution.log_prob(actions)
        ratios = torch.exp(log_probs - old_log_probs)
        surrogate_loss = (ratios * advantages).mean()
        return -surrogate_loss

    def _prepare_and_normalize_obs(self, obs: TensorDict) -> TensorDict:
        og_shape = dict()

        # assuming obs is a flat dict, collapse time and envs dimensions into a single batch dimension
        for key, x in obs.items():
            og_shape[key] = x.shape
            obs[key] = x.view((x.shape[0] * x.shape[1],) + x.shape[2:])

        # hold the lock while we alter the state of the normalizer since they can be used in other processes too
        with self.param_server.policy_lock:
            normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)

        # restore original shape
        for key, x in normalized_obs.items():
            normalized_obs[key] = x.view(og_shape[key])

        return normalized_obs

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = shallow_recursive_copy(batch)

            # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
            valids: Tensor = buff["policy_id"] == self.policy_id
            # ignore experience that was older than the threshold even before training started
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            # for last T+1 step, we want to use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]

            # ensure we're in train mode so that normalization statistics are updated
            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # don't need non-normalized obs anymore

            # calculate estimated value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            next_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)["values"]
            buff["values"][:, -1] = next_values

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                denormalized_values = buff["values"].clone()  # need to clone since normalizer is in-place
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values = buff["values"]

            # calculate advantage estimate
            buff["advantages"] = gae_advantages(
                buff["rewards"],
                buff["dones"],
                denormalized_values,
                buff["valids"],
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )
            # here returns are not normalized yet, so we should use denormalized values
            buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]

            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            # return normalization parameters are only used on the learner, no need to lock the mutex
            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])  # in-place

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(f"{self.policy_id=} batch has {invalid_fraction:.2%} of invalid samples")

                # invalid action values can cause problems when we calculate logprobs
                # here we set them to 0 just to be safe
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                # likewise, some invalid values of log_prob_actions can cause NaNs or infs
                buff["log_prob_actions"][invalid_indices] = -1  # -1 seems like a safe value

            return buff, dataset_size, num_invalids
