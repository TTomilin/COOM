import numpy as np
import tensorflow as tf
from typing import List, Tuple

from CL.methods.regularization import Regularization_SAC


class EWC_SAC(Regularization_SAC):
    """EWC regularization method.

    https://arxiv.org/abs/1612.00796"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @tf.function
    def _get_grads(
            self,
            obs: tf.Tensor,
            one_hot: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape(persistent=True) as g:
            # Main outputs from computation graph
            logits = self.actor(obs, one_hot)
            logits = tf.reduce_sum(logits, -1)

            q1 = self.critic1(obs, one_hot)
            q2 = self.critic2(obs, one_hot)

            q1 = tf.reduce_sum(q1, -1)
            q2 = tf.reduce_sum(q2, -1)

        # Compute diagonal of the Fisher matrix
        actor_gs = g.jacobian(logits, self.actor_common_variables)
        q1_gs = g.jacobian(q1, self.critic1.common_variables)
        q2_gs = g.jacobian(q2, self.critic2.common_variables)
        del g
        return actor_gs, q1_gs, q2_gs

    def _get_importance_weights(self, **batch) -> List[tf.Tensor]:
        actor_gs, q1_gs, q2_gs = self._get_grads(batch['obs'], batch['one_hot'])

        reg_weights = []
        for gs in actor_gs:
            if gs is None:
                raise ValueError("Actor gradients are None!")

            # Fisher information summing over the output dimensions
            fisher = tf.reduce_sum(gs**2, 1)

            # Clip from below
            fisher = tf.clip_by_value(fisher, 1e-5, np.inf)

            # Average over the examples in the batch
            reg_weights += [tf.reduce_mean(fisher, 0)]

        critic_coef = 1.0 if self.regularize_critic else 0.0
        for q_g in q1_gs:
            fisher = q_g**2
            reg_weights += [critic_coef * tf.reduce_mean(fisher, 0)]

        for q_g in q2_gs:
            fisher = q_g**2
            reg_weights += [critic_coef * tf.reduce_mean(fisher, 0)]

        return reg_weights
