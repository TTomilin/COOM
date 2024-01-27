import numpy as np
import random
import tensorflow as tf


class ExplorationHelper:
    def __init__(self, kind: str, num_available_heads: int, num_tasks: int):
        self.kind = kind
        self.num_available_heads = num_available_heads
        self.num_tasks = num_tasks
        self.current_head_id = None
        self.current_rewards = []
        self.episode_returns = [[] for _ in range(self.num_available_heads)]

    def update_reward(self, reward):
        # Pass relevant info from SAC algorithm after a step.
        assert self.current_head_id is not None
        self.current_rewards.append(reward)

    def _get_one_hot(self, x):
        # return tf.one_hot(x, self.num_available_heads).numpy()
        return tf.one_hot(x, self.num_tasks).numpy()

    def select(self, head):
        self.current_head_id = head
        return self._get_one_hot(head)

    def get_exploration_head_one_hot(self):
        assert (self.current_head_id is None) == (len(self.current_rewards) == 0)

        if self.current_head_id is not None:
            # Previous exploration trajectory has finished, collect statistics
            self.episode_returns[self.current_head_id].append(sum(self.current_rewards))
            self.current_rewards = []

        if self.kind == "current":
            return self.select(self.num_available_heads - 1)

        if self.kind == "previous":
            return self.select(self.num_available_heads - 2)

        if self.kind == "uniform_previous":
            return self.select(random.randint(0, self.num_available_heads - 2))

        if self.kind == "uniform_previous_or_current":
            return self.select(random.randint(0, self.num_available_heads - 1))

        # For other strategies: if some previous head is unused, return it
        for i in range(self.num_available_heads - 1):
            if len(self.episode_returns[i]) == 0:
                return self.select(i)

        if self.kind == "best_return":
            scores = []
            for i in range(self.num_available_heads - 1):
                score = np.mean(self.episode_returns[i])
                scores.append(score)
            chosen = int(np.argmax(scores))
            return self.select(chosen)
