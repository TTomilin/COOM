from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.ppo_learner import PPOLearner
from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.learning.trpo_learner import TRPOLearner
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import gae_advantages
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import log


class TRPOLagLearner(TRPOLearner):
    def __init__(
            self,
            cfg: Config,
            env_info: EnvInfo,
            policy_versions_tensor: Tensor,
            policy_id: PolicyID,
            param_server: ParameterServer,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.lambda_lagr = None

    def init(self) -> InitModelData:
        init_res = super().init()
        self.lambda_lagr = self.cfg.lambda_lagr
        return init_res

    def _calculate_losses(
            self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Tensor, Dict]:
        with torch.no_grad(), self.timing.add_time("losses_init"):
            valids = mb.valids

        # Calculate policy head outside of recurrent loop
        with self.timing.add_time("forward_head"):
            actor_head_outputs, critic_head_outputs, cost_critic_head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = actor_head_outputs.size(0)

        # Initial RNN states
        with self.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()

                # Split rnn_states into actor and critic components
                actor_rnn_states, critic_rnn_states, cost_critic_rnn_states = mb.rnn_states.chunk(3, dim=1)

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

                # Build RNN inputs for cost critic
                cost_critic_head_output_seq, cost_critic_rnn_states, cost_critic_inverted_inds = build_rnn_inputs(
                    cost_critic_head_outputs,
                    done_or_invalid,
                    cost_critic_rnn_states,
                    self.cfg.recurrence,
                )
            else:
                actor_rnn_states, critic_rnn_states, cost_critic_rnn_states = mb.rnn_states[::self.cfg.recurrence].chunk(3, dim=1)

        # Calculate RNN outputs for each timestep in a loop
        with self.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                with self.timing.add_time("bptt_forward_core"):
                    with torch.backends.cudnn.flags(enabled=False):
                        actor_core_output_seq, critic_core_output_seq, cost_critic_core_output_seq, _, _, _ = self.actor_critic.forward_core(
                            actor_head_output_seq, critic_head_output_seq, cost_critic_head_output_seq, actor_rnn_states, critic_rnn_states, cost_critic_rnn_states)
                actor_core_outputs = build_core_out_from_seq(actor_core_output_seq, actor_inverted_inds)
                critic_core_outputs = build_core_out_from_seq(critic_core_output_seq, critic_inverted_inds)
                cost_critic_core_outputs = build_core_out_from_seq(cost_critic_core_output_seq, cost_critic_inverted_inds)
                del actor_core_output_seq
                del critic_core_output_seq
                del cost_critic_core_output_seq
            else:
                actor_core_outputs, critic_core_outputs, cost_critic_core_outputs, _, _, _ = self.actor_critic.forward_core(actor_head_outputs, critic_head_outputs, cost_critic_head_outputs, actor_rnn_states, critic_rnn_states, cost_critic_rnn_states)

            del actor_head_outputs
            del critic_head_outputs
            del cost_critic_head_outputs

        assert actor_core_outputs.shape[0] == minibatch_size

        with self.timing.add_time("tail"):
            # Calculate policy tail outside of recurrent loop
            result = self.actor_critic.forward_tail(actor_core_outputs, critic_core_outputs, cost_critic_core_outputs, values_only=False,
                                                    sample_actions=False)
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)

            # For TRPO, compute the old action distribution for KL divergence
            with torch.no_grad():
                old_action_distribution = get_action_distribution(
                    self.actor_critic.action_space,
                    mb.action_logits,
                )

            values = result["values"].squeeze()
            cost_values = result["cost_values"].squeeze()

            del actor_core_outputs
            del critic_core_outputs
            del cost_critic_core_outputs

        # Lagrangian Update
        mean_cost = mb["costs"].mean()
        with self.timing.add_time("lagrange_update"):
            cost_violation = self._update_lagrange(mean_cost)

        # Compute advantages
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            # Using regular GAE
            adv = mb.advantages
            targets = mb.returns
            cost_adv = mb.cost_advantages

            # Compute modified advantage with Lagrangian term
            adv = self._compute_adv_surrogate(adv, cost_adv)
            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # Normalize advantage

        with self.timing.add_time("losses"):
            # Policy loss for TRPO
            masked_adv = masked_select(adv, valids, num_invalids)
            masked_log_prob_actions = masked_select(log_prob_actions, valids, num_invalids)
            policy_loss = -torch.mean(masked_adv * masked_log_prob_actions)

            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)

            # KL divergence between new and old policies
            kl_div = old_action_distribution.kl_divergence(action_distribution)
            kl_div = masked_select(kl_div, valids, num_invalids)
            kl_loss = kl_div.mean()

            # Critic losses (MSE)
            value_loss = self._value_loss(values, targets, valids, num_invalids)
            cost_value_loss = self._value_loss(cost_values, mb.cost_returns, valids, num_invalids)

        loss_summaries = dict(
            values=result["values"],
            cost_values=result["cost_values"],
            avg_cost=mean_cost,
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            kl_loss=kl_loss,
            cost_violation=cost_violation,
            lagrange_multiplier=self._get_lagrange_multiplier(),
        )

        return action_distribution, policy_loss, exploration_loss, value_loss, cost_value_loss, loss_summaries

    def _value_loss(self, values, targets, valids, num_invalids):
        masked_values = masked_select(values, valids, num_invalids)
        masked_targets = masked_select(targets, valids, num_invalids)
        return torch.nn.functional.mse_loss(masked_values, masked_targets)

    def _update_lagrange(self, mean_cost):
        # Calculate the average cost constraint violation
        cost_violation = (mean_cost - self.safety_bound).detach()
        # Update lambda_lagr based on the violation magnitude
        delta_lambda_lagr = cost_violation * self.cfg.lagrangian_coef_rate
        new_lambda_lagr = self.lambda_lagr + delta_lambda_lagr
        # Ensure lambda_lagr remains non-negative
        new_lambda_lagr = torch.nn.ReLU()(new_lambda_lagr)
        self.lambda_lagr = new_lambda_lagr
        return cost_violation

    def _get_lagrange_multiplier(self):
        return self.lambda_lagr

    def _compute_adv_surrogate(self, adv, cost_adv):
        return adv - self.lambda_lagr * cost_adv  # subtract cost advantages

    def _train(
            self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
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
                        cost_value_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    actor_loss = surrogate_loss + exploration_loss
                    total_loss = actor_loss + value_loss + cost_value_loss
                    epoch_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(total_loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f cvl:%.4f exp_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(total_loss),
                            to_scalar(surrogate_loss),
                            to_scalar(value_loss),
                            to_scalar(cost_value_loss),
                            to_scalar(exploration_loss),
                        )
                        force_summaries = True

                with timing.add_time("update"):
                    self._trpo_step(mb, actor_loss, value_loss + cost_value_loss, num_invalids, mb.valids)
                    num_optimization_steps += 1

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
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
                actor_head_outputs, critic_head_outputs, cost_critic_head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
                if self.cfg.use_rnn:
                    # Rebuild RNN inputs if necessary
                    done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()

                    # Split rnn_states into actor and critic components
                    actor_rnn_states, critic_rnn_states, cost_critic_rnn_states = mb.rnn_states.chunk(3, dim=1)

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

                    # Build RNN inputs for cost critic
                    cost_critic_head_output_seq, cost_critic_rnn_states, cost_critic_inverted_inds = build_rnn_inputs(
                        cost_critic_head_outputs,
                        done_or_invalid,
                        cost_critic_rnn_states,
                        self.cfg.recurrence,
                    )

                    with torch.backends.cudnn.flags(enabled=False):
                        actor_core_output_seq, critic_core_output_seq, cost_critic_core_output_seq, _, _, _ = self.actor_critic.forward_core(
                            actor_head_output_seq, critic_head_output_seq, cost_critic_head_output_seq, actor_rnn_states, critic_rnn_states, cost_critic_rnn_states)
                    actor_core_outputs = build_core_out_from_seq(actor_core_output_seq, actor_inverted_inds)
                    critic_core_outputs = build_core_out_from_seq(critic_core_output_seq, critic_inverted_inds)
                    cost_critic_core_outputs = build_core_out_from_seq(cost_critic_core_output_seq, cost_critic_inverted_inds)
                else:
                    actor_rnn_states, critic_rnn_states, cost_critic_rnn_states = mb.rnn_states[::self.cfg.recurrence].chunk(3, dim=1)
                    actor_core_outputs, critic_core_outputs, cost_critic_core_outputs, _, _, _ = self.actor_critic.forward_core(actor_head_outputs, critic_head_outputs, cost_critic_head_outputs, actor_rnn_states, critic_rnn_states, cost_critic_rnn_states)

                result = self.actor_critic.forward_tail(actor_core_outputs, critic_core_outputs, cost_critic_core_outputs, values_only=False,
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

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        var = train_loop_vars

        stats = super()._record_summaries(train_loop_vars)
        stats.cost_violation = var.cost_violation
        stats.avg_cost = var.avg_cost
        stats.lagrange_multiplier = var.lagrange_multiplier
        return stats

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # Create a shallow copy so we can modify the dictionary
            buff = shallow_recursive_copy(batch)

            # Ignore experience from other agents and from inactive agents
            valids: Tensor = buff["policy_id"] == self.policy_id
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            # For last T+1 step, we use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]

            # Ensure we're in train mode so that normalization statistics are updated
            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # Don't need non-normalized obs anymore

            # Calculate estimated value and cost value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            next_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)
            buff["values"][:, -1] = next_values["values"]
            buff["cost_values"][:, -1] = next_values["cost_values"]

            if self.cfg.normalize_returns:
                # Denormalize values for GAE calculation
                denormalized_values = buff["values"].clone()
                denormalized_cost_values = buff["cost_values"].clone()
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
                self.actor_critic.costs_normalizer(denormalized_cost_values, denormalize=True)
            else:
                denormalized_values = buff["values"]
                denormalized_cost_values = buff["cost_values"]

            # Calculate advantage estimates for rewards and costs
            buff["advantages"] = gae_advantages(
                buff["rewards"],
                buff["dones"],
                denormalized_values,
                buff["valids"],
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )

            buff["cost_advantages"] = gae_advantages(
                buff["costs"],
                buff["dones"],
                denormalized_cost_values,
                buff["valids"],
                self.cfg.gamma,
                self.cfg.gae_lambda,
            )

            # Compute returns
            buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]
            buff["cost_returns"] = buff["cost_advantages"] + buff["valids"][:, :-1] * denormalized_cost_values[:, :-1]

            # Remove next step obs, rnn_states, and values from the batch
            for key in ["normalized_obs", "rnn_states", "values", "cost_values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # Collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            # Normalize returns
            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])  # In-place normalization
                self.actor_critic.costs_normalizer(buff["cost_returns"])  # In-place normalization

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(f"{self.policy_id=} batch has {invalid_fraction:.2%} of invalid samples")

                # Set invalid action values to 0
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                buff["log_prob_actions"][invalid_indices] = -1  # Safe value

            return buff, dataset_size, num_invalids
