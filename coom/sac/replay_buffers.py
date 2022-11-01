import numpy as np
import random
import tensorflow as tf
from typing import Dict, Iterable


class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_shape: Iterable[int], size: int, num_tasks: int) -> None:
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

    def __init__(self, obs_shape: Iterable[int], size: int, num_tasks: int) -> None:
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

    def __init__(self, obs_shape: Iterable[int], size: int, num_tasks: int) -> None:
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


class Experience:
    def __init__(self, obs, action, reward, next_obs, done, one_hot):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.done = done
        self.one_hot = one_hot

    def __repr__(self):
        return f"Experience(obs={self.obs}, action={self.action}, reward={self.reward}, next_obs={self.next_obs}, done={self.done}, one_hot={self.one_hot})"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return self.obs == other.obs and self.action == other.action and self.reward == other.reward and self.next_obs == other.next_obs and self.done == other.done and self.one_hot == other.one_hot

    def __hash__(self):
        return hash((self.obs, self.action, self.reward, self.next_obs, self.done, self.one_hot))


class PrioritizedReplayBuffer(ReplayBuffer):
    PER_e = 0.01  # Avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Make a trade-off between random sampling and only taking high priority exp
    PER_b = 0.4  # Importance-sampling, from initial value increasing to 1
    PER_b_increment = 0.001  # Importance-sampling increment per sampling

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, obs_shape: Iterable[int], size: int, num_tasks: int) -> None:
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

        experience = Experience(obs, action, reward, next_obs, done, one_hot)

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

        return b_idx, np.array(memory_b), b_ISWeights

    def batch_update(self, tree_idx: np.ndarray, abs_errors: np.ndarray) -> None:
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
        tree_index  0 0  0  We fill the leaves from left to right
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
