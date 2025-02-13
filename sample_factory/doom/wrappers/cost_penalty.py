import gymnasium as gym


class CostPenalty(gym.Wrapper):
    """
    A gym environment wrapper that deducts cost from the reward.

    This wrapper modifies the reward returned by the environment by deducting a specified cost,
    which is a part of the `info` dictionary returned by the environment's `step` method.
    This approach is useful for incorporating safety constraints into reinforcement learning tasks,
    where the cost of an action can detract from its overall reward.

    """

    def __init__(self, env, penalty_scaling):
        super().__init__(env)
        self.penalty_scaling = penalty_scaling

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        cost = info['cost']
        reward -= cost * self.penalty_scaling
        return observation, reward, terminated, truncated, info
