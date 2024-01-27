from typing import Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Categorical

from CL.replay.buffers import EpisodicMemory, ReplayBuffer
from CL.rl.sac import SAC
from CL.utils.running import create_one_hot_vec


class ClonExSAC(SAC):
    def __init__(self, episodic_mem_per_task: int = 0, episodic_batch_size: int = 0, regularize_critic: bool = False,
                 cl_reg_coef: float = 0., episodic_memory_from_buffer: bool = True, **vanilla_sac_kwargs):
        """Episodic replay.

        Args:
          episodic_mem_per_task: Number of examples to keep in additional memory per task.
          episodic_batch_size: Minibatch size to compute additional loss.
        """
        super().__init__(**vanilla_sac_kwargs)

        self.episodic_mem_per_task = episodic_mem_per_task
        self.episodic_batch_size = episodic_batch_size
        self.regularize_critic = regularize_critic
        self.cl_reg_coef = cl_reg_coef
        self.episodic_memory_from_buffer = episodic_memory_from_buffer

        num_tasks = self.env.num_tasks
        episodic_mem_size = self.episodic_mem_per_task * num_tasks
        self.episodic_memory = EpisodicMemory(obs_shape=self.obs_shape, act_dim=self.act_dim, size=episodic_mem_size,
                                              num_tasks=num_tasks, save_targets=True)

    def behavioral_cloning_gradients(
            self,
            obs: tf.Tensor,
            actions: tf.Tensor,
            task_ids: tf.Tensor,
            target_actor_logits: tf.Tensor,
            target_critic1_preds: tf.Tensor,
            target_critic2_preds: tf.Tensor):

        with tf.GradientTape(persistent=True) as g:
            logits = self.actor(obs, task_ids)
            actor_loss_per_example = kl_divergence(target_actor_logits, logits)

            actor_loss = tf.reduce_mean(actor_loss_per_example)
            actor_loss *= self.cl_reg_coef

            if self.regularize_critic:
                critic1_pred = self.critic1(obs, task_ids)
                critic2_pred = self.critic2(obs, task_ids)

                critic1_loss_per_example = (critic1_pred - target_critic1_preds)**2
                critic2_loss_per_example = (critic2_pred - target_critic2_preds)**2

                critic1_loss = tf.reduce_mean(critic1_loss_per_example)
                critic2_loss = tf.reduce_mean(critic2_loss_per_example)
                critic_loss = critic1_loss + critic2_loss
                critic_loss *= self.cl_reg_coef

        actor_gradients = g.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(critic_loss, self.critic_variables) if self.regularize_critic else None

        return actor_gradients, critic_gradients, actor_loss

    def adjust_gradients(
            self,
            actor_gradients: List[tf.Tensor],
            critic_gradients: List[tf.Tensor],
            alpha_gradient: List[tf.Tensor],
            current_task_idx: int,
            metrics: dict,
            episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        if current_task_idx > 0:
            ref_actor_gradients, ref_critic_gradients, kl_loss = self.behavioral_cloning_gradients(
                obs=episodic_batch["obs"],
                actions=episodic_batch["actions"],
                task_ids=episodic_batch["one_hot"],
                target_actor_logits=episodic_batch["actor_logits"],
                target_critic1_preds=episodic_batch["critic1_preds"],
                target_critic2_preds=episodic_batch["critic2_preds"],
            )

            final_actor_gradients = self.merge_gradients(actor_gradients, ref_actor_gradients)
            final_critic_gradients = self.merge_gradients(critic_gradients, ref_critic_gradients)
            metrics["kl_loss"] = kl_loss
        else:
            final_actor_gradients = actor_gradients
            final_critic_gradients = critic_gradients

        return final_actor_gradients, final_critic_gradients, alpha_gradient

    def merge_gradients(self, new_grads: List[tf.Tensor], ref_grads: Optional[List[tf.Tensor]]):
        if ref_grads is None:
            return new_grads
        final_grads = []
        for new_grad, ref_grad in zip(new_grads, ref_grads):
            final_grads += [(new_grad + ref_grad) / 2]
        return final_grads

    def gather_buffer(self, task_idx):
        tmp_replay_buffer = ReplayBuffer(self.obs_shape, self.episodic_mem_per_task, self.num_tasks)
        one_hot_vec = create_one_hot_vec(self.env.num_tasks, self.env.task_id)
        env_to_gather = self.env.envs[task_idx]
        obs, _ = env_to_gather.reset()
        episode_len = 0
        for step_idx in range(self.episodic_mem_per_task):
            action = self.get_action(tf.convert_to_tensor(obs), tf.convert_to_tensor(one_hot_vec, dtype=tf.float32))
            action = action.numpy()[0]
            next_obs, reward, done, truncated, info = env_to_gather.step(action)

            episode_len += 1
            done_to_store = False if episode_len == self.max_episode_len else done
            tmp_replay_buffer.store(obs, action, reward, next_obs, done_to_store, one_hot_vec)

            if done:
                obs, _ = env_to_gather.reset()
                episode_len = 0
            else:
                obs = next_obs
        return tmp_replay_buffer.sample_batch(self.episodic_mem_per_task)

    def on_task_start(self, current_task_idx: int) -> None:
        super(ClonExSAC, self).on_task_start(current_task_idx)
        if current_task_idx > 0:
            if self.episodic_memory_from_buffer:
                new_episodic_mem = self.replay_buffer.sample_batch(self.episodic_mem_per_task)
            else:
                new_episodic_mem = self.gather_buffer(current_task_idx - 1)

            obs = new_episodic_mem["obs"]
            one_hot_task_ids = new_episodic_mem["one_hot"]
            logits = self.actor(obs, one_hot_task_ids)
            critic1_preds = self.critic1(obs, one_hot_task_ids)
            critic2_preds = self.critic2(obs, one_hot_task_ids)

            new_episodic_mem = {k: v.numpy() for k, v in new_episodic_mem.items()}
            new_episodic_mem["actor_logits"] = logits.numpy()
            new_episodic_mem["critic1_preds"] = critic1_preds.numpy()
            new_episodic_mem["critic2_preds"] = critic2_preds.numpy()

            self.episodic_memory.store_multiple(**new_episodic_mem)

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        return None if current_task_idx == 0 else self.episodic_memory.sample_batch(self.episodic_batch_size)


def kl_divergence(q_logits, p_logits):
    first_dist = Categorical(logits=q_logits)
    second_dist = Categorical(logits=p_logits)
    return tfp.distributions.kl_divergence(first_dist, second_dist, allow_nan_stats=True, name=None)
