from __future__ import annotations

import time
from itertools import chain
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import Tensor
from torchviz import make_dot

from sample_factory.algo.learning.ppo_learner import PPOLearner
from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.utils.action_distributions import TupleActionDistribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import gae_advantages
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.model.actor_critic import create_actor_critic, Actor
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import log


class CPOLearner(PPOLearner):
    def __init__(
            self,
            cfg: Config,
            env_info: EnvInfo,
            policy_versions_tensor: Tensor,
            policy_id: PolicyID,
            param_server: ParameterServer,
    ):
        PPOLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.critic_optimizer = None
        self.cost_critic_optimizer = None
        self.optimizers = []

    def init(self) -> InitModelData:
        init_res = super(CPOLearner, self).init()
        critic_params = list(self.actor_critic.critic.parameters())
        cost_critic_params = list(self.actor_critic.cost_critic.parameters())
        self.critic_optimizer = self.create_optimizer(critic_params)
        self.cost_critic_optimizer = self.create_optimizer(cost_critic_params)
        self.optimizers = [self.critic_optimizer, self.cost_critic_optimizer]
        self._apply_lr(self.curr_lr)
        return init_res

    def _apply_lr(self, lr: float) -> None:
        """Change the learning rate of the optimizers."""

        def optimizer_lr(opt):
            for group in opt.param_groups:
                return group["lr"]

        for optimizer in self.optimizers:
            if lr != optimizer_lr(optimizer):
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            if grad is None:
                continue
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten

    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            if hessian is None:
                continue
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten

    def flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten

    def update_model(self, model, new_params):
        index = 0
        # Iterate through the parameters of core and action_parameterization components
        for params in model.parameters():
            # Calculate the number of elements in the current parameter tensor
            params_length = params.numel()  # Using numel is slightly more idiomatic than len(params.view(-1))
            # Slice the new_params tensor to get the new values for the current parameter tensor
            new_param = new_params[index: index + params_length]
            # Reshape the new parameter slice to the shape of the existing parameter tensor
            new_param = new_param.view(params.size())
            # Copy the new parameter values into the existing parameter tensor
            params.data.copy_(new_param)
            # Increment the index by the number of elements in the current parameter tensor
            index += params_length

    def kl_divergence(self, mb, new_actor, old_actor):
        action_distribution = self.get_action_distribution(mb, new_actor)
        action_distribution_old = self.get_action_distribution(mb, old_actor)

        if isinstance(action_distribution, TupleActionDistribution):
            kl_divergence = 0
            for i in range(len(action_distribution.distributions)):
                kl_divergence += self.kl_divergence_single(
                    action_distribution.distributions[i], action_distribution_old.distributions[i]
                )
        else:
            kl_divergence = self.kl_divergence_single(action_distribution, action_distribution_old)
        return kl_divergence

    def conjugate_gradient(self, mb, b, nsteps, residual_tol=1e-10):
        x = torch.zeros(b.size()).to(device=self.device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for _ in range(nsteps):
            _Avp = self.fisher_vector_product(mb, p)
            alpha = rdotr / (torch.dot(p, _Avp) + 1e-8)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def fisher_vector_product(self, mb, p):
        p.detach()
        kl = self.kl_divergence(mb, new_actor=self.actor_critic.actor, old_actor=self.actor_critic.actor).mean()
        kl_grad = torch.autograd.grad(kl, self.actor_critic.actor.parameters(), create_graph=True, allow_unused=True)
        kl_grad = self.flat_grad(kl_grad)

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor_critic.actor.parameters(), allow_unused=True)
        kl_hessian_p = self.flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    def _calculate_losses(
            self, mb: AttrDict, num_invalids: int, experience_size: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Tensor, Dict]:
        with torch.no_grad(), self.timing.add_time("losses_init"):

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

        action_distribution, num_trajectories, ratio, result, rnn_states = self.eval_actions(mb, self.actor_critic)

        values = result["values"].squeeze()
        cost_values = result["cost_values"].squeeze()
        valids = mb.valids
        recurrence: int = self.cfg.recurrence

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            if self.cfg.with_vtrace:
                # V-trace parameters
                rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                c_hat = torch.Tensor([self.cfg.vtrace_c])

                ratios_cpu = ratio.cpu()
                values_cpu = values.cpu()
                rewards_cpu = mb.rewards_cpu
                dones_cpu = mb.dones_cpu

                vtrace_rho = torch.min(rho_hat, ratios_cpu)
                vtrace_c = torch.min(c_hat, ratios_cpu)

                vs = torch.zeros((num_trajectories * recurrence))
                adv = torch.zeros((num_trajectories * recurrence))

                next_values = values_cpu[recurrence - 1:: recurrence] - rewards_cpu[recurrence - 1:: recurrence]
                next_values /= self.cfg.gamma
                next_vs = next_values

                for i in reversed(range(self.cfg.recurrence)):
                    rewards = rewards_cpu[i::recurrence]
                    dones = dones_cpu[i::recurrence]
                    not_done = 1.0 - dones
                    not_done_gamma = not_done * self.cfg.gamma

                    curr_values = values_cpu[i::recurrence]
                    curr_vtrace_rho = vtrace_rho[i::recurrence]
                    curr_vtrace_c = vtrace_c[i::recurrence]

                    delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                    adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                    next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                    vs[i::recurrence] = next_vs

                    next_values = curr_values

                targets = vs.to(self.device)
                adv = adv.to(self.device)
                # TODO implement cost V-trace
            else:
                # using regular GAE
                adv = mb.advantages
                targets = mb.returns
                cost_adv = mb.cost_advantages
                cost_targets = mb.cost_returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            cost_adv_std, cost_adv_mean = torch.std_mean(masked_select(cost_adv, valids, num_invalids))
            cost_adv = (cost_adv - cost_adv_mean) / torch.clamp_min(cost_adv_std, 1e-7)  # normalize cost advantage

        with self.timing.add_time("exploration_loss"):
            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)

        with self.timing.add_time("kl_loss"):
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )

        with self.timing.add_time("critic_loss"):
            old_values = mb["values"]
            old_cost_values = mb["cost_values"]
            value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            cost_loss = self._value_loss(cost_values, old_cost_values, cost_targets, clip_value, valids, num_invalids)

        # Visualize the graph
        dot = make_dot(values)
        dot.render('values', format='png')
        dot = make_dot(cost_values)
        dot.render('cost_values', format='png')

        with self.timing.add_time("critic_update"):
            # Following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
            for p in chain(self.actor_critic.critic.parameters(), self.actor_critic.cost_critic.parameters()):
                p.grad = None
            value_loss.backward()
            cost_loss.backward()

            if self.cfg.max_grad_norm > 0.0:
                with self.timing.add_time("clip"):
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.cost_critic.parameters(), self.cfg.max_grad_norm)

            actual_lr = self.curr_lr
            if num_invalids > 0:
                # if we have masked (invalid) data we should reduce the learning rate accordingly
                # this prevents a situation where most of the data in the minibatch is invalid
                # and we end up doing SGD with super noisy gradients
                actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
            self._apply_lr(actual_lr)

            with self.param_server.policy_lock:
                self.critic_optimizer.step()
                self.cost_critic_optimizer.step()

        with self.timing.add_time("policy_loss"):
            rescale_constraint_val = (mb["costs"].mean() - self.env_info.safety_bound) * (1 - self.cfg.gamma)
            if rescale_constraint_val == 0:
                rescale_constraint_val = 1e-8

            reward_loss = -torch.sum(ratio * adv, dim=-1, keepdim=True).mean()
            reward_loss_grad = torch.autograd.grad(reward_loss, self.actor_critic.actor.parameters(), retain_graph=True,
                                                   allow_unused=True)
            reward_loss_grad = self.flat_grad(reward_loss_grad)

            cost_loss = torch.sum(ratio * cost_adv, dim=-1, keepdim=True).mean()
            cost_loss_grad = torch.autograd.grad(cost_loss, self.actor_critic.actor.parameters(), retain_graph=True,
                                                 allow_unused=True)
            cost_loss_grad = self.flat_grad(cost_loss_grad)

            B_cost_loss_grad = cost_loss_grad.unsqueeze(0)
            B_cost_loss_grad = self.flat_grad(B_cost_loss_grad)

            g_step_dir = self.conjugate_gradient(mb, reward_loss_grad.data, nsteps=10)
            b_step_dir = self.conjugate_gradient(mb, B_cost_loss_grad.data, nsteps=10)

            q_coef = (reward_loss_grad * g_step_dir).sum(0, keepdim=True)
            r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)
            s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)

            fraction = self.cfg.line_search_fraction
            loss_improve = 0

            B_cost_loss_grad_dot = torch.dot(B_cost_loss_grad, B_cost_loss_grad)
            if (torch.dot(B_cost_loss_grad, B_cost_loss_grad)) <= 1e-8 and rescale_constraint_val < 0:
                b_step_dir = torch.tensor(0)
                r_coef = torch.tensor(0)
                s_coef = torch.tensor(0)
                positive_Cauchy_value = torch.tensor(0)
                whether_recover_policy_value = torch.tensor(0)
                optim_case = 4
            else:
                r_coef = (reward_loss_grad * b_step_dir).sum(0, keepdim=True)
                s_coef = (cost_loss_grad * b_step_dir).sum(0, keepdim=True)
                if r_coef == 0:
                    r_coef = 1e-8
                if s_coef == 0:
                    s_coef = 1e-8
                positive_Cauchy_value = (
                        q_coef - (r_coef**2) / (1e-8 + s_coef))
                whether_recover_policy_value = 2 * self.cfg.kl_threshold - (
                        rescale_constraint_val**2) / (
                                                       1e-8 + s_coef)
                if rescale_constraint_val < 0 and whether_recover_policy_value < 0:
                    optim_case = 3
                elif rescale_constraint_val < 0 and whether_recover_policy_value >= 0:
                    optim_case = 2
                elif rescale_constraint_val >= 0 and whether_recover_policy_value >= 0:
                    optim_case = 1
                else:
                    optim_case = 0

            if whether_recover_policy_value == 0:
                whether_recover_policy_value = 1e-8

            if optim_case in [3, 4]:
                lam = torch.sqrt(
                    (q_coef / (2 * self.cfg.kl_threshold)))
                nu = torch.tensor(0)  # v_coef = 0
            elif optim_case in [1, 2]:
                LA, LB = [0, r_coef / rescale_constraint_val], [r_coef / rescale_constraint_val, np.inf]
                LA, LB = (LA, LB) if rescale_constraint_val < 0 else (LB, LA)
                proj = lambda x, L: max(L[0], min(L[1], x))
                lam_a = proj(torch.sqrt(positive_Cauchy_value / whether_recover_policy_value), LA)
                lam_b = proj(torch.sqrt(q_coef / (torch.tensor(2 * self.cfg.kl_threshold))), LB)

                f_a = lambda lam: -0.5 * (positive_Cauchy_value / (
                        1e-8 + lam) + whether_recover_policy_value * lam) - r_coef * rescale_constraint_val / (
                                          1e-8 + s_coef)
                f_b = lambda lam: -0.5 * (q_coef / (1e-8 + lam) + 2 * self.cfg.kl_threshold * lam)
                lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
                nu = max(0, lam * rescale_constraint_val - r_coef) / (1e-8 + s_coef)
            else:
                lam = torch.tensor(0)
                nu = torch.sqrt(torch.tensor(2 * self.cfg.kl_threshold) / (1e-8 + s_coef))

            x_a = (1. / (lam + 1e-8)) * (g_step_dir + nu * b_step_dir)
            x_b = (nu * b_step_dir)
            x = x_a if optim_case > 0 else x_b

            reward_loss = reward_loss.detach()
            cost_loss = cost_loss.detach()
            params = self.flat_params(self.actor_critic)

            old_actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
            old_actor_critic.model_to_device(self.device)
            self.update_model(old_actor_critic, params)

            expected_improve = -torch.dot(x, reward_loss_grad).sum(0, keepdim=True)
            expected_improve = expected_improve.detach()

            flag = False
            fraction_coef = self.cfg.fraction_coef
            for i in range(self.cfg.ls_step):
                x_norm = torch.norm(x)
                if x_norm > 0.5:
                    x = x * 0.5 / x_norm

                actor_params = self.flat_params(self.actor_critic.actor)
                new_params = actor_params - fraction_coef * (fraction**i) * x
                self.update_model(self.actor_critic.actor, new_params)
                action_distribution, num_trajectories, ratio, result, rnn_states = self.eval_actions(mb,
                                                                                                     self.actor_critic)

                new_reward_loss = torch.sum(ratio * adv, dim=-1, keepdim=True).mean()
                new_cost_loss = torch.sum(ratio * cost_adv, dim=-1, keepdim=True).mean()

                new_reward_loss = new_reward_loss.detach()
                new_reward_loss = -new_reward_loss
                new_cost_loss = new_cost_loss.detach()
                loss_improve = new_reward_loss - reward_loss

                kl = self.kl_divergence(mb, new_actor=self.actor_critic.actor, old_actor=old_actor_critic.actor).mean()

                if ((kl < self.cfg.kl_threshold) and (loss_improve < 0 if optim_case > 1 else True)
                        and (new_cost_loss.mean() - cost_loss.mean() <= max(-rescale_constraint_val, 0))):
                    flag = True
                    break
                expected_improve *= fraction

            if not flag:
                params = self.flat_params(old_actor_critic)
                self.update_model(self.actor_critic.actor, params)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=result["values"],
            cost_values=result["cost_values"],
            avg_cost=mb["costs"].mean(),
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            learning_rate=self.curr_lr,
            kl_divergence=kl.item(),
            improvement_ratio=(loss_improve / expected_improve).item() if expected_improve != 0 else float('inf'),
            policy_update_acceptance=flag,
            cost_metric_change=(new_cost_loss.mean() - cost_loss.mean()).item(),
            step_size=fraction_coef * (fraction**i),
            convergence_metric=torch.norm(new_params - actor_params).item(),
            policy_entropy=torch.distributions.Categorical(logits=result['action_logits']).entropy().mean().item(),
            line_search_steps=i,
        )

        return action_distribution, reward_loss, exploration_loss, kl_old, kl_loss, value_loss, cost_loss, loss_summaries

    def get_action_distribution(self, mb, actor: Actor):
        with self.timing.add_time("forward_actor"):
            rnn_states = mb.rnn_states[::self.cfg.recurrence]
            _, action_distribution = actor.forward(mb.normalized_obs, rnn_states)
        return action_distribution

    def eval_actions(self, mb, actor_critic):
        # calculate policy head outside of recurrent loop
        with self.timing.add_time("forward_head"):
            head_outputs = actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = head_outputs[0].size(0)
        # initial rnn states
        with self.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                # this is the only way to stop RNNs from backpropagating through invalid timesteps
                # (i.e. experience collected by another policy)
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~mb.valids.cpu()).float()
                concat_head_output = torch.cat(head_outputs, dim=1)
                head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                    concat_head_output,
                    done_or_invalid,
                    mb.rnn_states,
                    self.cfg.recurrence,
                )
            else:
                rnn_states = mb.rnn_states[::self.cfg.recurrence]

        # calculate RNN outputs for each timestep in a loop
        with self.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                with self.timing.add_time("bptt_forward_core"):
                    core_output_seq, _ = actor_critic.forward_core(head_output_seq, rnn_states)
                if isinstance(core_output_seq, List):
                    core_outputs = []
                    for core_output in core_output_seq:
                        core_outputs.append(build_core_out_from_seq(core_output, inverted_select_inds))
                else:
                    core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
                del core_output_seq
            else:
                core_outputs = self.actor_critic.forward_core(head_outputs, rnn_states)

            del head_outputs

        num_trajectories = minibatch_size // self.cfg.recurrence
        assert core_outputs[0].shape[0] == minibatch_size

        with self.timing.add_time("tail"):
            # calculate policy tail outside of recurrent loop
            result = actor_critic.forward_tail(core_outputs, values_only=False, sample_actions=False)

            del core_outputs

            action_distribution = actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)
        return action_distribution, num_trajectories, ratio, result, rnn_states

    def _train(
            self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                        self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
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

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        cost_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids, experience_size)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss
                    critic_loss = value_loss
                    loss: Tensor = actor_loss + critic_loss + cost_loss

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                # with torch.no_grad(), timing.add_time("kl_divergence"):
                #     # if kl_old is not None it is already calculated above
                #     if kl_old is None:
                #         # calculate KL-divergence with the behaviour policy action distribution
                #         old_action_distribution = get_action_distribution(
                #             self.actor_critic.action_space,
                #             mb.action_logits,
                #         )
                #         kl_old = action_distribution.kl_divergence(old_action_distribution)
                #         kl_old = masked_select(kl_old, mb.valids, num_invalids)
                #
                #     kl_old_mean = float(kl_old.mean().item())
                #     recent_kls.append(kl_old_mean)
                #     if kl_old.numel() > 0 and kl_old.max().item() > 100:
                #         log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                # update the weights
                # with timing.add_time("update"):
                # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                # for p in self.actor_critic.parameters():
                #     p.grad = None
                #
                # loss.backward()

                # if self.cfg.max_grad_norm > 0.0:
                #     with timing.add_time("clip"):
                #         torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                #
                #
                # actual_lr = self.curr_lr
                # if num_invalids > 0:
                #     # if we have masked (invalid) data we should reduce the learning rate accordingly
                #     # this prevents a situation where most of the data in the minibatch is invalid
                #     # and we end up doing SGD with super noisy gradients
                #     actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                # self._apply_lr(actual_lr)

                # with self.param_server.policy_lock:
                #     self.optimizer.step()

                curr_policy_version = self.train_step  # policy version before the weight update
                num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        var = train_loop_vars
        params = self.actor_critic.parameters()
        grad_norm = (sum(p.grad.data.norm(2).item()**2 for p in params if p.grad is not None)**0.5)
        self.last_summary_time = time.time()

        stats = AttrDict()
        stats.lr = self.curr_lr
        stats.update(self.actor_critic.summaries())
        stats.valids_fraction = var.mb.valids.float().mean()
        stats.same_policy_fraction = (var.mb.policy_id == self.policy_id).float().mean()
        stats.grad_norm = grad_norm
        stats.loss = var.loss
        stats.value = var.values.mean()
        stats.entropy = var.action_distribution.entropy().mean()
        stats.policy_loss = var.policy_loss
        stats.kl_loss = var.kl_loss
        stats.value_loss = var.value_loss
        stats.exploration_loss = var.exploration_loss
        stats.cost_loss = var.cost_loss
        stats.cost_values = var.cost_values.mean()
        stats.avg_cost = var.avg_cost
        stats.learning_rate = var.learning_rate
        stats.kl_divergence = var.kl_divergence
        stats.improvement_ratio = var.improvement_ratio
        stats.policy_update_acceptance = var.policy_update_acceptance
        stats.cost_metric_change = var.cost_metric_change
        stats.step_size = var.step_size
        stats.convergence_metric = var.convergence_metric
        stats.policy_entropy = var.policy_entropy
        stats.line_search_steps = var.line_search_steps

        stats.act_min = var.mb.actions.min()
        stats.act_max = var.mb.actions.max()

        if "adv_mean" in stats:
            stats.adv_min = var.mb.advantages.min()
            stats.adv_max = var.mb.advantages.max()
            stats.adv_std = var.adv_std
            stats.adv_mean = var.adv_mean

        stats.max_abs_logprob = torch.abs(var.mb.action_logits).max()

        if hasattr(var.action_distribution, "summaries"):
            stats.update(var.action_distribution.summaries())

        if var.epoch == self.cfg.num_epochs - 1 and var.batch_num == len(var.minibatches) - 1:
            # we collect these stats only for the last PPO batch, or every time if we're only doing one batch, IMPALA-style
            valid_ratios = masked_select(var.ratio, var.mb.valids, var.num_invalids)
            ratio_mean = torch.abs(1.0 - valid_ratios).mean().detach()
            ratio_min = valid_ratios.min().detach()
            ratio_max = valid_ratios.max().detach()
            # log.debug('Learner %d ratio mean min max %.4f %.4f %.4f', self.policy_id, ratio_mean.cpu().item(), ratio_min.cpu().item(), ratio_max.cpu().item())

            value_delta = torch.abs(var.values - var.mb.values)
            value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

            stats.value_delta = value_delta_avg
            stats.value_delta_max = value_delta_max
            # noinspection PyUnresolvedReferences
            stats.fraction_clipped = (
                    (valid_ratios < var.clip_ratio_low).float() + (valid_ratios > var.clip_ratio_high).float()
            ).mean()
            stats.ratio_mean = ratio_mean
            stats.ratio_min = ratio_min
            stats.ratio_max = ratio_max
            stats.num_sgd_steps = var.num_sgd_steps

        version_diff = (var.curr_policy_version - var.mb.policy_version)[var.mb.policy_id == self.policy_id]
        stats.version_diff_avg = version_diff.mean()
        stats.version_diff_min = version_diff.min()
        stats.version_diff_max = version_diff.max()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

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
            next_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)
            buff["values"][:, -1] = next_values["values"]
            buff["cost_values"][:, -1] = next_values["cost_values"]

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                # rl_games PPO uses a similar approach, see:
                # https://github.com/Denys88/rl_games/blob/7b5f9500ee65ae0832a7d8613b019c333ecd932c/rl_games/algos_torch/models.py#L51
                denormalized_values = buff["values"].clone()  # need to clone since normalizer is in-place
                denormalized_cost_values = buff["cost_values"].clone()  # need to clone since normalizer is in-place
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
                self.actor_critic.costs_normalizer(denormalized_cost_values, denormalize=True)
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values = buff["values"]
                denormalized_cost_values = buff["cost_values"]

            if self.cfg.value_bootstrap:
                # Value bootstrapping is a technique that reduces the surprise for the critic in case
                # we're ending the episode by timeout. Intuitively, in this case the cumulative return for the last step
                # should not be zero, but rather what the critic expects. This improves learning in many envs
                # because otherwise the critic cannot predict the abrupt change in rewards in a timed-out episode.
                # What we really want here is v(t+1) which we don't have because we don't have obs(t+1) (since
                # the episode ended). Using v(t) is an approximation that requires that rew(t) can be generally ignored.

                # Multiply by both time_out and done flags to make sure we count only timeouts in terminal states.
                # There was a bug in older versions of isaacgym where timeouts were reported for non-terminal states.
                buff["rewards"].add_(self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"])

            if not self.cfg.with_vtrace:
                # calculate advantage estimate (in case of V-trace it is done separately for each minibatch)
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

                # here returns are not normalized yet, so we should use denormalized values
                buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]
                buff["cost_returns"] = buff["cost_advantages"] + buff["valids"][:, :-1] * denormalized_cost_values[:,
                                                                                          :-1]

            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values", "cost_values", "valids"]:
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
                self.actor_critic.costs_normalizer(buff["cost_returns"])  # in-place

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

    def kl_divergence_single(self, dist_new, dist_old):
        return torch.sum(dist_old.probs * (dist_old.log_probs - dist_new.log_probs), dim=-1, keepdim=True)
