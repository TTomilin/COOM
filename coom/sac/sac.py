import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras.optimizers.schedules.learning_rate_schedule import LearningRateSchedule
from tensorflow.python.framework import dtypes
from tensorflow_probability.python.distributions import Categorical

from coom.env.scenario.common import CommonEnv
from coom.sac import models
from coom.sac.models import PopArtMlpCritic
from coom.sac.replay_buffers import ReplayBuffer, ReservoirReplayBuffer
from coom.sac.utils.logx import EpochLogger
from coom.utils.enums import BufferType
from coom.utils.utils import reset_optimizer, reset_weights, set_seed


class SAC:
    def __init__(
            self,
            env: CommonEnv,
            test_envs: List[CommonEnv],
            logger: EpochLogger,
            scenarios: List[str],
            cl_method: str = None,
            actor_cl: type = models.MlpActor,
            critic_cl: type = models.MlpCritic,
            policy_kwargs: Dict = None,
            seed: int = 0,
            steps_per_env: int = 4e5,
            log_every: int = 1000,
            replay_size: int = 1e5,
            gamma: float = 0.99,
            polyak: float = 0.995,
            lr: float = 1e-3,
            lr_decay: str = None,
            lr_decay_rate: float = 0.1,
            lr_decay_steps: int = None,
            alpha: Union[float, str] = "auto",
            batch_size: int = 128,
            start_steps: int = 1e3,
            update_after: int = 1000,
            update_every: int = 50,
            num_test_eps_stochastic: int = 10,
            num_test_eps_deterministic: int = 1,
            save_freq_epochs: int = 25,
            reset_buffer_on_task_change: bool = True,
            buffer_type: BufferType = BufferType.FIFO,
            reset_optimizer_on_task_change: bool = False,
            reset_critic_on_task_change: bool = False,
            clipnorm: float = None,
            target_output_std: float = None,
            agent_policy_exploration: bool = False,
            render: bool = False,
            render_sleep: float = 0.03,
            experiment_dir: Path = None,
            model_path: str = None,
            timestamp: str = None,
    ):
        """A class for SAC training, for either single task, continual learning or multi-task learning.
        After the instance is created, use run() function to actually run the training.

        Args:
          env: An environment on which training will be performed.
          test_envs: Environments on which evaluation will be periodically performed;
            for example, when env is a multi-task environment, test_envs can be a list of individual
            task environments.
          logger: An object for logging the results.
          actor_cl: Class for actor model.
          actor_kwargs: Kwargs for actor model.
          critic_cl: Class for critic model.
          critic_kwargs: Kwargs for critic model.
          seed: Seed for randomness.
          steps_per_env: Number of steps the algorithm will run per environment.
          log_every: Number of steps between subsequent evaluations and logging.
          replay_size: Size of the replay buffer.
          gamma: Discount factor.
          polyak: Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:
              target_weights <- polyak * target_weights + (1 - polyak) * weights
            (Always between 0 and 1, usually close to 1.)
          lr: Learning rate for the optimizer.
          alpha: Entropy regularization coefficient. Can be either float value,
            or "auto", in which case it is dynamically tuned.
            (Equivalent to inverse of reward scale in the original SAC paper.)
          batch_size: Minibatch size for the optimization.
          start_steps: Number of steps for uniform-random action selection, before running real
            policy. Helps exploration.
          update_after: Number of env interactions to collect before starting to do gradient
            descent updates.  Ensures replay buffer is full enough for useful updates.
          update_every: Number of env interactions that should elapse between gradient descent
            updates.
            Note: Regardless of how long you wait between updates, the ratio of env steps to
            gradient steps is locked to 1.
          num_test_eps_stochastic: Number of episodes to test the stochastic policy in each
            evaluation.
          num_test_eps_deterministic: Number of episodes to test the deterministic policy in each
            evaluation.
          save_freq_epochs: How often, in epochs, to save the current policy and value function.
            (Epoch is defined as time between two subsequent evaluations, lasting log_every steps)
          reset_buffer_on_task_change: If True, replay buffer will be cleared after every task
            change (in continual learning).
          buffer_type: Type of the replay buffer. Either 'fifo' for regular FIFO buffer
            or 'reservoir' for reservoir sampling.
          reset_optimizer_on_task_change: If True, optimizer will be reset after every task change
            (in continual learning).
          reset_critic_on_task_change: If True, critic weights are randomly re-initialized after
            each task change.
          clipnorm: Value for gradient clipping.
          target_output_std: If alpha is 'auto', alpha is dynamically tuned so that standard
            deviation of the action distribution on every dimension matches target_output_std.
          agent_policy_exploration: If True, uniform exploration for start_steps steps is used only
            in the first task (in continual learning). Otherwise, it is used in every task.
        """
        set_seed(seed, env=env)

        if policy_kwargs is None:
            policy_kwargs = {}

        self.env = env
        self.num_tasks = env.num_tasks
        self.test_envs = test_envs
        self.logger = logger
        self.scenarios = scenarios
        self.cl_method = cl_method
        self.critic_cl = critic_cl
        self.policy_kwargs = policy_kwargs
        self.steps_per_env = steps_per_env
        self.steps = steps_per_env * env.num_tasks
        self.log_every = log_every
        self.replay_size = replay_size
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_test_eps_stochastic = num_test_eps_stochastic
        self.num_test_eps_deterministic = num_test_eps_deterministic
        self.save_freq_epochs = save_freq_epochs
        self.reset_buffer_on_task_change = reset_buffer_on_task_change
        self.buffer_type = buffer_type
        self.reset_optimizer_on_task_change = reset_optimizer_on_task_change
        self.reset_critic_on_task_change = reset_critic_on_task_change
        self.clipnorm = clipnorm
        self.agent_policy_exploration = agent_policy_exploration
        self.render = render
        self.render_sleep = render_sleep
        self.experiment_dir = experiment_dir
        self.model_path = model_path
        self.timestamp = timestamp

        self.use_popart = critic_cl is PopArtMlpCritic

        self.obs_shape = env.observation_space.shape
        self.act_dim = env.action_space.n
        print("Observations shape:", self.obs_shape)  # should be N_FRAMES x H x W
        print("Actions shape:", self.act_dim)

        # Share environment information with the policy architecture
        policy_kwargs["state_space"] = env.observation_space
        policy_kwargs["action_space"] = env.action_space
        policy_kwargs["num_tasks"] = env.num_tasks

        # Create experience buffer
        if buffer_type == BufferType.FIFO:
            self.replay_buffer = ReplayBuffer(
                obs_shape=self.obs_shape, size=replay_size, num_tasks=self.num_tasks
            )
        elif buffer_type == BufferType.RESERVOIR:
            self.replay_buffer = ReservoirReplayBuffer(
                obs_shape=self.obs_shape, size=replay_size, num_tasks=self.num_tasks
            )

        # Create actor and critic networks
        self.actor = actor_cl(**policy_kwargs)

        self.critic1 = critic_cl(**policy_kwargs)
        self.target_critic1 = critic_cl(**policy_kwargs)
        self.target_critic1.set_weights(self.critic1.get_weights())

        self.critic2 = critic_cl(**policy_kwargs)
        self.target_critic2 = critic_cl(**policy_kwargs)
        self.target_critic2.set_weights(self.critic2.get_weights())

        if model_path is not None:
            self.load_model(model_path)

        self.critic_variables = self.critic1.trainable_variables + self.critic2.trainable_variables
        self.all_common_variables = (
                self.actor.common_variables
                + self.critic1.common_variables
                + self.critic2.common_variables
        )

        # Learning rate schedule
        if lr_decay_steps is None:
            lr_decay_steps = steps_per_env
        if lr_decay == 'exponential':
            lr = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=lr_decay_steps,
                decay_rate=lr_decay_rate)
        elif lr_decay == 'linear':
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=lr,
                decay_steps=lr_decay_steps,
                end_learning_rate=lr * lr_decay_rate,
                power=1.0,
                cycle=False,
                name=None
            )

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # For reference on automatic alpha tuning, see
        # "Automating Entropy Adjustment for Maximum Entropy" section
        # in https://arxiv.org/abs/1812.05905
        self.auto_alpha = False
        if alpha == "auto":
            self.auto_alpha = True
            self.all_log_alpha = tf.Variable(
                np.ones((self.num_tasks, 1), dtype=np.float32), trainable=True
            )
            if target_output_std is None:
                self.target_entropy = -np.prod(env.action_space.shape).astype(np.float32)
            else:
                target_1d_entropy = np.log(target_output_std * math.sqrt(2 * math.pi * math.e))
                self.target_entropy = (
                        np.prod(env.action_space.n).astype(np.float32) * target_1d_entropy
                )

    def adjust_gradients(
            self,
            actor_gradients: List[tf.Tensor],
            critic_gradients: List[tf.Tensor],
            alpha_gradient: List[tf.Tensor],
            current_task_idx: int,
            metrics: dict,
            episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        return actor_gradients, critic_gradients, alpha_gradient

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        return tf.constant(0.0)

    def on_test_start(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_test_end(self, seq_idx: tf.Tensor) -> None:
        pass

    def on_task_start(self, current_task_idx: int) -> None:
        print(f'Task {current_task_idx} started')

    def on_task_end(self, current_task_idx: int) -> None:
        print(f'Task {current_task_idx} finished')

    def get_episodic_batch(self, current_task_idx: int) -> Optional[Dict[str, tf.Tensor]]:
        return None

    def get_log_alpha(self, one_hot: tf.Tensor) -> tf.Tensor:
        return tf.squeeze(tf.linalg.matmul(tf.expand_dims(tf.convert_to_tensor(one_hot), 1), self.all_log_alpha))

    @tf.function
    def get_action(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor,
                   deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        logits = self.actor(tf.expand_dims(obs, 0), tf.expand_dims(one_hot_task_id, 0))

        dist = Categorical(logits=logits)
        if deterministic:
            return tf.math.argmax(logits, axis=-1, output_type=dtypes.int32)
        else:
            return dist.sample()

    def get_action_test(
            self, obs: tf.Tensor, one_hot_task_id: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)
    ) -> tf.Tensor:
        return self.get_action(obs, one_hot_task_id, deterministic).numpy()[0]

    def get_learn_on_batch(self, current_task_idx: int) -> Callable:
        @tf.function
        def learn_on_batch(
                seq_idx: tf.Tensor,
                batch: Dict[str, tf.Tensor],
                episodic_batch: Dict[str, tf.Tensor] = None,
        ) -> Dict:
            gradients, metrics = self.get_gradients(seq_idx, **batch)
            # Warning: we refer here to the int task_idx in the parent function, not
            # the passed seq_idx.
            gradients = self.adjust_gradients(
                *gradients,
                current_task_idx=current_task_idx,
                metrics=metrics,
                episodic_batch=episodic_batch,
            )

            if self.clipnorm is not None:
                actor_gradients, critic_gradients, alpha_gradient = gradients
                gradients = (
                    tf.clip_by_global_norm(actor_gradients, self.clipnorm)[0],
                    tf.clip_by_global_norm(critic_gradients, self.clipnorm)[0],
                    tf.clip_by_norm(alpha_gradient, self.clipnorm),
                )

            self.apply_update(*gradients)
            return metrics

        return learn_on_batch

    def get_gradients(
            self,
            seq_idx: tf.Tensor,
            obs: tf.Tensor,
            next_obs: tf.Tensor,
            actions: tf.Tensor,
            rewards: tf.Tensor,
            done: tf.Tensor,
            one_hot: tf.Tensor,
    ) -> Tuple[Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Dict]:
        with tf.GradientTape(persistent=True) as g:
            if self.auto_alpha:
                log_alpha = self.get_log_alpha(one_hot)
            else:
                log_alpha = tf.math.log(self.alpha)
            log_alpha_exp = tf.math.exp(log_alpha)

            logits = self.actor(obs, one_hot)
            dist = Categorical(logits=logits)
            entropy = dist.entropy()

            logits_next = self.actor(next_obs, one_hot)
            dist_next = Categorical(logits=logits_next)
            entropy_next = dist_next.entropy()

            q1 = self.critic1(obs, one_hot)
            q2 = self.critic2(obs, one_hot)

            # Q values of actions taken
            q1_vals = tf.gather(q1, actions, axis=1, batch_dims=1)
            q2_vals = tf.gather(q2, actions, axis=1, batch_dims=1)

            # Target Q values
            target_q1 = self.target_critic1(next_obs, one_hot)
            target_q2 = self.target_critic2(next_obs, one_hot)

            # Min Double-Q:
            min_q = dist.probs_parameter() * tf.stop_gradient(tf.minimum(q1, q2))
            min_target_q = dist_next.probs_parameter() * tf.minimum(target_q1, target_q2)

            # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
            if self.critic_cl is PopArtMlpCritic:
                q_backup = tf.stop_gradient(
                    self.critic1.normalize(
                        rewards + self.gamma * (1 - done)
                        * (self.critic1.unnormalize(min_target_q, next_obs) - log_alpha_exp * entropy_next),
                        obs,
                    )
                )
            else:
                q_backup = tf.stop_gradient(
                    rewards + self.gamma * (1 - done)
                    * (tf.math.reduce_sum(min_target_q, axis=-1) - log_alpha_exp * entropy_next)
                )

            # Critic loss
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_vals)**2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_vals)**2)
            value_loss = q1_loss + q2_loss

            # Actor loss
            actor_loss = -tf.reduce_mean(log_alpha_exp * entropy + tf.reduce_sum(min_q, axis=-1))

            # Alpha loss
            if self.auto_alpha:
                log_prob = tf.stop_gradient(entropy) + self.target_entropy
                alpha_loss = -tf.reduce_mean(log_alpha * log_prob)

            auxiliary_loss = self.get_auxiliary_loss(seq_idx)
            metrics = dict(
                pi_loss=actor_loss,
                q1_loss=q1_loss,
                q2_loss=q2_loss,
                q1=q1_vals,
                q2=q2_vals,
                entropy=entropy,
                reg_loss=auxiliary_loss,
                agem_violation=0,
            )

            actor_loss += auxiliary_loss
            value_loss += auxiliary_loss

        # Compute gradients
        actor_gradients = g.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(value_loss, self.critic_variables)
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g

        if self.use_popart:
            # Stats are shared between critic1 and critic2.
            # We keep them only in critic1.
            self.critic1.update_stats(q_backup, obs)

        gradients = (actor_gradients, critic_gradients, alpha_gradient)
        return gradients, metrics

    def apply_update(
            self,
            actor_gradients: List[tf.Tensor],
            critic_gradients: List[tf.Tensor],
            alpha_gradient: List[tf.Tensor],
    ) -> None:
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        self.optimizer.apply_gradients(zip(critic_gradients, self.critic_variables))

        if self.auto_alpha:
            self.optimizer.apply_gradients([(alpha_gradient, self.all_log_alpha)])

        # Polyak averaging for target variables
        for v, target_v in zip(
                self.critic1.trainable_variables, self.target_critic1.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
        for v, target_v in zip(
                self.critic2.trainable_variables, self.target_critic2.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)

    def test_agent(self, deterministic, num_episodes) -> None:
        mode = "deterministic" if deterministic else "stochastic"
        for seq_idx, test_env in enumerate(self.test_envs):
            key_prefix = f"test/{mode}/{seq_idx}/{test_env.name}"
            one_hot_vec = create_one_hot_vec(test_env.num_tasks, test_env.task_id)

            self.on_test_start(seq_idx)

            for j in range(num_episodes):
                obs, done, episode_return, episode_len = test_env.reset(), False, 0, 0
                while not done:
                    obs, reward, done, _ = test_env.step(
                        self.get_action_test(tf.convert_to_tensor(obs),
                                             tf.convert_to_tensor(one_hot_vec, dtype=tf.dtypes.float32),
                                             tf.constant(deterministic))
                    )
                    episode_return += reward
                    episode_len += 1
                self.logger.store({key_prefix + "/return": episode_return, key_prefix + "/ep_length": episode_len})
                self.logger.store(test_env.get_statistics(key_prefix))

            self.on_test_end(seq_idx)

            self.logger.log_tabular(key_prefix + "/return", with_min_and_max=True)
            self.logger.log_tabular(key_prefix + "/ep_length", average_only=True)
            for stat in test_env.get_statistics(key_prefix).keys():
                self.logger.log_tabular(stat, average_only=True)

    def _log_after_update(self, results):
        self.logger.store(
            {
                "train/q1_vals": results["q1"],
                "train/q2_vals": results["q2"],
                "train/entropy": results["entropy"],
                "train/loss_pi": results["pi_loss"],
                "train/loss_q1": results["q1_loss"],
                "train/loss_q2": results["q2_loss"],
                "train/loss_reg": results["reg_loss"],
                "train/agem_violation": results["agem_violation"],
            }
        )

        for task_idx in range(self.num_tasks):
            if self.auto_alpha:
                self.logger.store(
                    {f"train/alpha/{task_idx}": float(tf.math.exp(self.all_log_alpha[task_idx][0]))}
                )
            if self.use_popart:
                self.logger.store(
                    {
                        f"train/popart_mean/{task_idx}": self.critic1.moment1[task_idx][0],
                        f"train/popart_std/{task_idx}": self.critic1.sigma[task_idx][0],
                    }
                )

    def _log_after_epoch(self, epoch, current_task_timestep, global_timestep, info, learning_rate):
        # Log info about epoch
        self.logger.log_tabular("epoch", epoch)
        self.logger.log_tabular("learning_rate", learning_rate)
        self.logger.log_tabular("train/return", with_min_and_max=True)
        self.logger.log_tabular("train/ep_length", average_only=True)
        self.logger.log_tabular("total_env_steps", global_timestep + 1)
        self.logger.log_tabular("current_task_steps", current_task_timestep + 1)
        self.logger.log_tabular("train/q1_vals", with_min_and_max=True)
        self.logger.log_tabular("train/q2_vals", with_min_and_max=True)
        self.logger.log_tabular("train/entropy", with_min_and_max=True)
        self.logger.log_tabular("train/loss_pi", average_only=True)
        self.logger.log_tabular("train/loss_q1", average_only=True)
        self.logger.log_tabular("train/loss_q2", average_only=True)
        for task_idx in range(self.num_tasks):
            if self.auto_alpha:
                self.logger.log_tabular(f"train/alpha/{task_idx}", average_only=True)
            if self.use_popart:
                self.logger.log_tabular(f"train/popart_mean/{task_idx}", average_only=True)
                self.logger.log_tabular(f"train/popart_std/{task_idx}", average_only=True)
        self.logger.log_tabular("train/loss_reg", average_only=True)
        for stat in self.env.get_statistics('train').keys():
            self.logger.log_tabular(stat, average_only=True)

        if "seq_idx" in info:
            self.logger.log_tabular("train/active_env", info["seq_idx"])

        self.logger.log_tabular("walltime", time.time() - self.start_time)
        self.logger.dump_tabular()

    def save_model(self, current_task_idx):
        method = self.cl_method if self.cl_method else "sac"
        model_dir = f'{self.experiment_dir}/checkpoints/{method}/{self.timestamp}/{self.env.task}'
        dir_prefixes = []
        if current_task_idx == -1:
            dir_prefixes.append(model_dir)
        else:
            dir_prefixes.append(f"{model_dir}_task{current_task_idx}")
            if current_task_idx == self.num_tasks - 1:
                dir_prefixes.append(model_dir)

        for prefix in dir_prefixes:
            self.actor.save_weights(os.path.join(prefix, "actor"))
            self.critic1.save_weights(os.path.join(prefix, "critic1"))
            self.target_critic1.save_weights(os.path.join(prefix, "target_critic1"))
            self.critic2.save_weights(os.path.join(prefix, "critic2"))
            self.target_critic2.save_weights(os.path.join(prefix, "target_critic2"))

    def load_model(self, model_path):
        checkpoint_dir = f'{self.experiment_dir}/checkpoints/{model_path}'
        self.actor.load_weights(os.path.join(checkpoint_dir, "actor"))
        self.critic1.load_weights(os.path.join(checkpoint_dir, "critic1"))
        self.target_critic1.load_weights(os.path.join(checkpoint_dir, "target_critic1"))
        self.critic2.load_weights(os.path.join(checkpoint_dir, "critic2"))
        self.target_critic2.load_weights(os.path.join(checkpoint_dir, "target_critic2"))

    def _handle_task_change(self, current_task_idx: int):
        self.on_task_start(current_task_idx)

        if self.reset_buffer_on_task_change:
            assert self.buffer_type == BufferType.FIFO
            self.replay_buffer = ReplayBuffer(
                obs_shape=self.obs_shape, size=self.replay_size, num_tasks=self.num_tasks
            )
        if self.reset_critic_on_task_change:
            reset_weights(self.critic1, self.critic_cl, self.policy_kwargs)
            self.target_critic1.set_weights(self.critic1.get_weights())
            reset_weights(self.critic2, self.critic_cl, self.policy_kwargs)
            self.target_critic2.set_weights(self.critic2.get_weights())

        if self.reset_optimizer_on_task_change:
            reset_optimizer(self.optimizer)

        # Update variables list and update function in case model changed.
        # E.g: For VCL after the first task we set trainable=False for layer
        # normalization. We need to recompute the graph in order for TensorFlow
        # to notice this change.
        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)
        self.all_common_variables = (
                self.actor.common_variables
                + self.critic1.common_variables
                + self.critic2.common_variables
        )

    def run(self):
        """A method to run the SAC training, after the object has been created."""
        self.start_time = time.time()
        obs, episode_return, episode_len = self.env.reset(), 0, 0

        # Main loop: collect experience in env and update/log each epoch
        current_task_timestep = 0
        current_task_idx = -1
        self.learn_on_batch = self.get_learn_on_batch(current_task_idx)

        for global_timestep in range(self.steps):
            # On task change
            if current_task_idx != getattr(self.env, "cur_seq_idx", -1):
                current_task_timestep = 0
                current_task_idx = getattr(self.env, "cur_seq_idx")
                self._handle_task_change(current_task_idx)

            # Until start_steps have elapsed, randomly sample actions from a uniform
            # distribution for better exploration. Afterwards, use the learned policy.
            if current_task_timestep > self.start_steps or (self.agent_policy_exploration and current_task_idx > 0):
                one_hot_vec = create_one_hot_vec(self.env.num_tasks, self.env.task_id)
                action = self.get_action(tf.convert_to_tensor(obs),
                                         tf.convert_to_tensor(one_hot_vec, dtype=tf.dtypes.float32)).numpy()[0]
            else:
                action = self.env.action_space.sample()

            # Step the env
            next_obs, reward, done, info = self.env.step(action)
            episode_return += reward
            episode_len += 1

            if self.render:
                time.sleep(self.render_sleep)

            # Extract task ids from the info dict
            one_hot_vec = create_one_hot_vec(self.env.num_tasks, self.env.task_id)

            # Store experience to replay buffer
            self.replay_buffer.store(obs, action, reward, next_obs, done, one_hot_vec)

            # Update the most recent observation
            obs = next_obs

            # End of trajectory handling
            if done:
                self.logger.store({"train/return": episode_return, "train/ep_length": episode_len})
                self.logger.store(self.env.get_statistics('train'))
                self.env.clear_episode_statistics()
                episode_return, episode_len = 0, 0
                if global_timestep < self.steps - 1:
                    obs = self.env.reset()

            # Update handling
            if current_task_timestep >= self.update_after and current_task_timestep % self.update_every == 0:

                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.batch_size)

                    episodic_batch = self.get_episodic_batch(current_task_idx)

                    results = self.learn_on_batch(
                        tf.convert_to_tensor(current_task_idx), batch, episodic_batch
                    )
                    self._log_after_update(results)

            if self.env.name == "ContinualLearningEnv" and current_task_timestep + 1 == self.env.steps_per_env:
                self.on_task_end(current_task_idx)

            # End of epoch wrap-up
            if ((global_timestep + 1) % self.log_every == 0) or (global_timestep + 1 == self.steps):
                epoch = (global_timestep + 1 + self.log_every - 1) // self.log_every

                # Save model
                if (epoch % self.save_freq_epochs == 0) or (global_timestep + 1 == self.steps):
                    self.save_model(current_task_idx)

                # Test the performance of stochastic and deterministic version of the agent.
                self.test_agent(deterministic=False, num_episodes=self.num_test_eps_stochastic)
                self.test_agent(deterministic=True, num_episodes=self.num_test_eps_deterministic)

                # Determine the current learning rate of the optimizer
                lr = self.optimizer.lr
                if issubclass(type(lr), LearningRateSchedule):
                    lr = self.optimizer._decayed_lr('float32').numpy()

                self._log_after_epoch(epoch, current_task_timestep, global_timestep, info, lr)

            current_task_timestep += 1


def create_one_hot_vec(num_tasks, task_id):
    one_hot_vec = np.zeros(num_tasks)
    one_hot_vec[task_id] = 1.0
    return one_hot_vec
