import numpy as np
import random
import tensorflow as tf
from enum import Enum
from typing import Dict, Tuple, Optional
from typing import Union

from cl.sac.tree import SumTree, SegmentTree


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"
    PRIORITY = "priority"
    PER = "per"


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_shape: Optional[Tuple[int, ...]], size: int, num_tasks: int) -> None:
        self.obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.actions_buf = np.zeros(size, dtype=np.int32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.one_hot_buf = np.zeros([size, num_tasks], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
            self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool,
            one_hot: np.ndarray
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.one_hot_buf[self.ptr] = one_hot
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            one_hot=tf.convert_to_tensor(self.one_hot_buf[idxs])
        )


class EpisodicMemory:
    """Buffer which does not support overwriting old samples."""

    def __init__(self, obs_shape: Optional[Tuple[int, ...]], act_dim: int, size: int, num_tasks: int,
                 save_targets: bool = False) -> None:
        self.obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.actions_buf = np.zeros(size, dtype=np.int32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.one_hot_buf = np.zeros([size, num_tasks], dtype=np.float32)
        self.size, self.max_size = 0, size
        self.save_targets = save_targets
        if self.save_targets:
            self.actor_logits_buf = np.zeros([size, act_dim], dtype=np.float32)
            self.critic1_pred_buf = np.zeros([size, act_dim], dtype=np.float32)
            self.critic2_pred_buf = np.zeros([size, act_dim], dtype=np.float32)

    def store_multiple(
            self,
            obs: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_obs: np.ndarray,
            done: np.ndarray,
            one_hot: np.ndarray,
            **kwargs: Dict[str, np.ndarray]
    ) -> None:
        assert len(obs) == len(actions) == len(rewards) == len(next_obs) == len(done)
        assert self.size + len(obs) <= self.max_size

        range_start = self.size
        range_end = self.size + len(obs)
        self.obs_buf[range_start:range_end] = obs
        self.next_obs_buf[range_start:range_end] = next_obs
        self.actions_buf[range_start:range_end] = actions
        self.rewards_buf[range_start:range_end] = rewards
        self.done_buf[range_start:range_end] = done
        self.one_hot_buf[range_start:range_end] = one_hot
        if self.save_targets:
            self.actor_logits_buf[range_start:range_end] = kwargs['actor_logits']
            self.critic1_pred_buf[range_start:range_end] = kwargs['critic1_preds']
            self.critic2_pred_buf[range_start:range_end] = kwargs['critic2_preds']
        self.size = self.size + len(obs)

    def sample_batch(self, batch_size: int, task_weights: Optional[np.ndarray] = None) -> Dict[str, tf.Tensor]:
        batch_size = min(batch_size, self.size)
        if task_weights is not None:
            task_ids = self.one_hot_buf[:self.size]
            example_weights = task_weights[task_ids]
            example_weights /= example_weights.sum()
            idxs = np.random.choice(self.size, size=batch_size, replace=False, p=example_weights)
        else:
            idxs = np.random.choice(self.size, size=batch_size, replace=False)
        batch_dict = dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            one_hot=tf.convert_to_tensor(self.one_hot_buf[idxs])
        )

        if self.save_targets:
            batch_dict["actor_logits"] = tf.convert_to_tensor(self.actor_logits_buf[idxs])
            batch_dict["critic1_preds"] = tf.convert_to_tensor(self.critic1_pred_buf[idxs])
            batch_dict["critic2_preds"] = tf.convert_to_tensor(self.critic2_pred_buf[idxs])

        return batch_dict


class ReservoirReplayBuffer(ReplayBuffer):
    """Buffer for SAC agents implementing reservoir sampling."""

    def __init__(self, obs_shape: Optional[Tuple[int, ...]], size: int, num_tasks: int) -> None:
        super().__init__(obs_shape, size, num_tasks)
        self.timestep = 0

    def store(
            self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool,
            one_hot: np.ndarray
    ) -> None:
        current_t = self.timestep
        self.timestep += 1

        if current_t < self.max_size:
            buffer_idx = current_t
        else:
            buffer_idx = random.randint(0, current_t)
            if buffer_idx >= self.max_size:
                return

        self.obs_buf[buffer_idx] = obs
        self.next_obs_buf[buffer_idx] = next_obs
        self.actions_buf[buffer_idx] = action
        self.rewards_buf[buffer_idx] = reward
        self.done_buf[buffer_idx] = done
        self.one_hot_buf[buffer_idx] = one_hot
        self.size = min(self.size + 1, self.max_size)


class PrioritizedReplayBuffer(ReplayBuffer):
    PER_e = 0.01  # Avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Make a trade-off between random sampling and only taking high priority exp
    PER_b = 0.4  # Importance-sampling, from initial value increasing to 1
    PER_b_increment = 0.001  # Importance-sampling increment per sampling

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, obs_shape: Optional[Tuple[int, ...]], size: int, num_tasks: int) -> None:
        super().__init__(obs_shape, size, num_tasks)
        self.buffer = SumTree(size)

    def store(
            self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool,
            one_hot: np.ndarray
    ) -> None:
        """
        Store the transitions of the designated previous n steps in a replay buffer
        Pop the leftmost transitions as the oldest in case the experience replay capacity is breached
        In case of prioritized replay find the max priority of the SumTree and add the experiences
        to the tree buffer with that priority value
        """
        # Find the maximum priority of the tree
        max_priority = np.max(self.buffer.tree[-self.buffer.capacity:])

        # Use minimum priority if the priority is 0, otherwise this experience will never have a chance to be selected
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        experience = (obs, next_obs, action, reward, done, one_hot)

        # Add the new experience to the tree with the maximum priority
        self.buffer.add(max_priority, experience)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        # Create a sample array that will contain the mini-batch
        memory_b = []

        # Create placeholders for the tree indexes and importance sampling weights
        b_idx = np.empty((batch_size,), dtype=np.int32)
        b_ISWeights = np.empty((batch_size, 1), dtype=np.float32)

        # Calculate the priority segment

        # Divide the Range[0, p_total] into n ranges
        priority_segment = self.buffer.total_priority / batch_size  # Priority segment

        # Increase the PER_b each time a new mini-batch is sampled
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment])  # Max = 1

        # Calculate the max_weight. Set it to a small value to avoid division by zero
        p_min = np.min(self.buffer.tree[-self.buffer.capacity:]) / self.buffer.total_priority
        max_weight = 1e-7 if p_min == 0 else (p_min * batch_size)**(-self.PER_b)

        for i in range(batch_size):
            """
            Uniformly sample a value from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that corresponds to each value that is retrieved
            """
            index, priority, data = self.buffer.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.buffer.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(batch_size * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            memory_b.append(data)

        memory_b = np.array(memory_b)
        batch = dict(
            obs=tf.convert_to_tensor(memory_b[:, 0].tolist(), dtype=tf.float32),
            next_obs=tf.convert_to_tensor(memory_b[:, 1].tolist(), dtype=tf.float32),
            actions=tf.convert_to_tensor(memory_b[:, 2].tolist(), dtype=tf.int32),
            rewards=tf.convert_to_tensor(memory_b[:, 3].tolist(), dtype=tf.float32),
            done=tf.convert_to_tensor(memory_b[:, 4].tolist(), dtype=tf.float32),
            one_hot=tf.convert_to_tensor(memory_b[:, 5].tolist(), dtype=tf.float32),
            idxs=tf.convert_to_tensor(b_idx, dtype=tf.int32),
            weights=tf.convert_to_tensor(b_ISWeights, dtype=tf.float32)
        )
        return batch

    def update_weights(self, tree_idx: np.ndarray, abs_errors: tf.Tensor) -> None:
        """
        Update the priorities in the tree
        """
        abs_errors += self.PER_e  # Avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.buffer.update(ti, p)

    @property
    def buffer_size(self):
        """
        Retrieve the number of gathered experience
        :return: Current size of the buffer
        """
        return self.buffer.data_pointer


class PrioritizedExperienceReplay(ReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.
    :param bool weight_norm: whether to normalize returned weights with the maximum
        weight value within the batch. Default to True.
    """

    def __init__(
            self,
            obs_shape: Optional[Tuple[int, ...]],
            size: int,
            num_tasks: int,
            alpha: float = 0.6,
            beta: float = 0.4,
            weight_norm: bool = True,
    ) -> None:
        ReplayBuffer.__init__(self, obs_shape, size, num_tasks)
        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        self.absolute_error_upper = 1.  # clipped abs error
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self._weight_norm = weight_norm

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio**self._alpha

    def store(
            self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool,
            one_hot: np.ndarray
    ) -> None:
        super().store(obs, action, reward, next_obs, done, one_hot)
        self.init_weight(self.ptr)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        scalar = np.random.rand(batch_size) * self.weight.reduce()
        idxs = self.weight.get_prefix_sum_idx(scalar)
        weight = self.get_weight(idxs)
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/memory.py L154
        weight = weight / np.max(weight) if self._weight_norm else weight
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            one_hot=tf.convert_to_tensor(self.one_hot_buf[idxs]),
            idxs=tf.convert_to_tensor(idxs),
            weights=tf.convert_to_tensor(weight),
        )

    def get_weight(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Get the importance sampling weight.

        The "weight" in the returned Batch is the weight on loss function to debias
        the sampling process (some transition tuples are sampled more often so their
        losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio)**(-self._beta)

    def update_weights(self, index: np.ndarray, new_weight: Union[np.ndarray, tf.Tensor]) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(np.array(new_weight, dtype=np.float64)) + self.__eps
        self.weight[index] = weight**self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def set_beta(self, beta: float) -> None:
        self._beta = beta
