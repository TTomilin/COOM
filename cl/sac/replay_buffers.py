import numpy as np
import random
import tensorflow as tf
from enum import Enum
from numba import njit
from typing import Dict, List, Tuple, Any
from typing import Optional, Union


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"
    PRIORITY = "priority"


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_shape: Tuple[int, int, int], size: int, num_tasks: int) -> None:
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

    def __init__(self, obs_shape: Tuple[int, int, int], size: int, num_tasks: int) -> None:
        self.obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_shape], dtype=np.float32)
        self.actions_buf = np.zeros(size, dtype=np.int32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.one_hot_buf = np.zeros([size, num_tasks], dtype=np.float32)
        self.size, self.max_size = 0, size

    def store_multiple(
            self,
            obs: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_obs: np.ndarray,
            done: np.ndarray,
            one_hot: np.ndarray,
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
        self.size = self.size + len(obs)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        batch_size = min(batch_size, self.size)
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
            one_hot=tf.convert_to_tensor(self.one_hot_buf[idxs])
        )


class ReservoirReplayBuffer(ReplayBuffer):
    """Buffer for SAC agents implementing reservoir sampling."""

    def __init__(self, obs_shape: Tuple[int, int, int], size: int, num_tasks: int) -> None:
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


class PrioritizedReplayBufferTianshou(ReplayBuffer):
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952.

    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.
    :param bool weight_norm: whether to normalize returned weights with the maximum
        weight value within the batch. Default to True.

    .. seealso::

        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(
            self,
            size: int,
            alpha: float,
            beta: float,
            weight_norm: bool = True,
            **kwargs: Any
    ) -> None:
        # will raise KeyError in PrioritizedVectorReplayBuffer
        # super().__init__(size, **kwargs)
        ReplayBuffer.__init__(self, size, **kwargs)
        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        # save weight directly in this class instead of self._meta
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self.options.update(alpha=alpha, beta=beta)
        self._weight_norm = weight_norm

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio**self._alpha

    def update(self, buffer: ReplayBuffer) -> np.ndarray:
        indices = super().update(buffer)
        self.init_weight(indices)
        return indices

    def add(
            self,
            batch: Any,
            buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ptr, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids)
        self.init_weight(ptr)
        return ptr, ep_rew, ep_len, ep_idx

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size > 0 and len(self) > 0:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)  # type: ignore
        else:
            return super().sample_indices(batch_size)

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

    def update_weight(
            self, index: np.ndarray, new_weight: Union[np.ndarray, tf.Tensor]
    ) -> None:
        """Update priority weight by index in this buffer.

        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(np.array(new_weight)) + self.__eps
        self.weight[index] = weight**self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Any:
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = self.sample_indices(0) if index == slice(None) \
                else self._indices[:len(self)][index]
        else:
            indices = index  # type: ignore
        batch = super().__getitem__(indices)
        weight = self.get_weight(indices)
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/memory.py L154
        batch.weight = weight / np.max(weight) if self._weight_norm else weight
        return batch

    def set_beta(self, beta: float) -> None:
        self._beta = beta


class PrioritizedReplayBuffer(ReplayBuffer):
    PER_e = 0.01  # Avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Make a trade-off between random sampling and only taking high priority exp
    PER_b = 0.4  # Importance-sampling, from initial value increasing to 1
    PER_b_increment = 0.001  # Importance-sampling increment per sampling

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, obs_shape: Tuple[int, int, int], size: int, num_tasks: int) -> None:
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

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, Dict[str, tf.Tensor], np.ndarray]:
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
        )
        return b_idx, batch, b_ISWeights

    def batch_update(self, tree_idx: np.ndarray, abs_errors: tf.Tensor) -> None:
        """
        Update the priorities in the tree
        """
        abs_errors += self.PER_e  # convert to abs and avoid 0
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


class SumTree(object):
    """
    This SumTree is a modified version of the implementation by Morvan Zhou:
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Initialize the nodes and data of the tree with zeros
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains [capacity] experiences
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Determine the index where the experience will be stored
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  We fill the leaves from left to right
        """

        # Update the data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Increment the data_pointer
        self.data_pointer += 1

        # Return to first index and start overwriting, if the capacity is breached
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index: int, priority: float) -> None:
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate the change through tree
        while tree_index != 0:  # This is faster than recursively looping

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES

                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 

            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class SegmentTree:
    """Implementation of Segment Tree.

    The segment tree stores an array ``arr`` with size ``n``. It supports value
    update and fast query of the sum for the interval ``[left, right)`` in
    O(log n) time. The detailed procedure is as follows:

    1. Pad the array to have length of power of 2, so that leaf nodes in the \
    segment tree have the same depth.
    2. Store the segment tree in a binary heap.

    :param int size: the size of segment tree.
    """

    def __init__(self, size: int) -> None:
        bound = 1
        while bound < size:
            bound *= 2
        self._size = size
        self._bound = bound
        self._value = np.zeros([bound * 2])
        self._compile()

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Return self[index]."""
        return self._value[index + self._bound]

    def __setitem__(
            self, index: Union[int, np.ndarray], value: Union[float, np.ndarray]
    ) -> None:
        """Update values in segment tree.

        Duplicate values in ``index`` are handled by numpy: later index
        overwrites previous ones.
        ::

            >>> a = np.array([1, 2, 3, 4])
            >>> a[[0, 1, 0, 1]] = [4, 5, 6, 7]
            >>> print(a)
            [6 7 3 4]
        """
        if isinstance(index, int):
            index, value = np.array([index]), np.array([value])
        assert np.all(0 <= index) and np.all(index < self._size)
        _setitem(self._value, index + self._bound, value)

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        """Return operation(value[start:end])."""
        if start == 0 and end is None:
            return self._value[1]
        if end is None:
            end = self._size
        if end < 0:
            end += self._size
        return _reduce(self._value, start + self._bound - 1, end + self._bound)

    def get_prefix_sum_idx(self, value: Union[float,
                                              np.ndarray]) -> Union[int, np.ndarray]:
        r"""Find the index with given value.

        Return the minimum index for each ``v`` in ``value`` so that
        :math:`v \le \mathrm{sums}_i`, where
        :math:`\mathrm{sums}_i = \sum_{j = 0}^{i} \mathrm{arr}_j`.

        .. warning::

            Please make sure all of the values inside the segment tree are
            non-negative when using this function.
        """
        assert np.all(value >= 0.0) and np.all(value < self._value[1])
        single = False
        if not isinstance(value, np.ndarray):
            value = np.array([value])
            single = True
        index = _get_prefix_sum_idx(value, self._bound, self._value)
        return index.item() if single else index

    def _compile(self) -> None:
        f64 = np.array([0, 1], dtype=np.float64)
        f32 = np.array([0, 1], dtype=np.float32)
        i64 = np.array([0, 1], dtype=np.int64)
        _setitem(f64, i64, f64)
        _setitem(f64, i64, f32)
        _reduce(f64, 0, 1)
        _get_prefix_sum_idx(f64, 1, f64)
        _get_prefix_sum_idx(f32, 1, f64)


@njit
def _setitem(tree: np.ndarray, index: np.ndarray, value: np.ndarray) -> None:
    """Numba version, 4x faster: 0.1 -> 0.024."""
    tree[index] = value
    while index[0] > 1:
        index //= 2
        tree[index] = tree[index * 2] + tree[index * 2 + 1]


@njit
def _reduce(tree: np.ndarray, start: int, end: int) -> float:
    """Numba version, 2x faster: 0.009 -> 0.005."""
    # nodes in (start, end) should be aggregated
    result = 0.0
    while end - start > 1:  # (start, end) interval is not empty
        if start % 2 == 0:
            result += tree[start + 1]
        start //= 2
        if end % 2 == 1:
            result += tree[end - 1]
        end //= 2
    return result


@njit
def _get_prefix_sum_idx(value: np.ndarray, bound: int, sums: np.ndarray) -> np.ndarray:
    """Numba version (v0.51), 5x speed up with size=100000 and bsz=64.

    vectorized np: 0.0923 (numpy best) -> 0.024 (now)
    for-loop: 0.2914 -> 0.019 (but not so stable)
    """
    index = np.ones(value.shape, dtype=np.int64)
    while index[0] < bound:
        index *= 2
        lsons = sums[index]
        direct = lsons < value
        value -= lsons * direct
        index += direct
    index -= bound
    return index
