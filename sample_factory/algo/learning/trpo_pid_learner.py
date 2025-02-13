from __future__ import annotations

import torch
from torch import Tensor

from sample_factory.algo.learning.pid_lagrangian import PIDLagrangian
from sample_factory.algo.learning.trpo_lag_learner import TRPOLagLearner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.utils.typing import Config, PolicyID


class TRPOPidLearner(TRPOLagLearner):
    def __init__(
            self,
            cfg: Config,
            env_info: EnvInfo,
            policy_versions_tensor: Tensor,
            policy_id: PolicyID,
            param_server: ParameterServer,
    ):
        TRPOLagLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.pid_lagrangian = PIDLagrangian(pid_kp=cfg.pid_kp, pid_ki=cfg.pid_ki, pid_kd=cfg.pid_kd,
                                            pid_d_delay=cfg.pid_d_delay,
                                            pid_delta_p_ema_alpha=cfg.pid_delta_p_ema_alpha,
                                            pid_delta_d_ema_alpha=cfg.pid_delta_d_ema_alpha, sum_norm=cfg.sum_norm,
                                            diff_norm=cfg.diff_norm, penalty_max=cfg.penalty_max,
                                            lagrangian_multiplier_init=cfg.lagrangian_multiplier_init,
                                            cost_limit=self.safety_bound)

    def _update_lagrange(self, mean_cost):
        return self.pid_lagrangian.pid_update(mean_cost)

    def _get_lagrange_multiplier(self):
        return self.pid_lagrangian.lagrangian_multiplier

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        CPPOPID uses the following surrogate loss:

        .. math::
            L = \frac{1}{1 + \lambda} [A^{R}_{\pi_{\theta}} (s, a) - \lambda A^C_{\pi_{\theta}} (s, a)]
        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The ``advantage`` combined with ``reward_advantage`` and ``cost_advantage``.
        """
        penalty = self.pid_lagrangian.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
