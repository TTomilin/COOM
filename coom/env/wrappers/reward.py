import numpy as np
from gym import RewardWrapper
from typing import Callable
from vizdoom import GameVariable


class WrapperHolder:

    def __init__(self, wrapper_class, **kwargs):
        self.wrapper_class = wrapper_class
        self.kwargs = kwargs


class ConstantRewardWrapper(RewardWrapper):

    def __init__(self, env, reward: float):
        super(ConstantRewardWrapper, self).__init__(env)
        self.rew = reward

    def reward(self, reward):
        return reward + self.rew


class BooleanVariableRewardWrapper(RewardWrapper):

    def __init__(self, env, reward: float, game_var: GameVariable):
        super(BooleanVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.game_var = game_var

    def reward(self, reward):
        game_variable = self.env.game.get_game_variable(self.game_var)
        if game_variable:
            reward += self.rew
        return reward


class GameVariableRewardWrapper(RewardWrapper):

    def __init__(self, env, reward: float, var_index: int = 0, decrease: bool = False):
        super(GameVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_index = var_index
        self.decrease = decrease

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward
        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        var_cur = vars_cur[self.var_index]
        var_prev = vars_prev[self.var_index]

        if not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            reward += self.rew
        return reward


class CumulativeVariableRewardWrapper(RewardWrapper):

    def __init__(self, env, reward: float, var_index: int = 0, decrease: bool = False, maintain: bool = False):
        super(CumulativeVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.var_index = var_index
        self.decrease = decrease
        self.maintain = maintain
        self.cum_rew = 0

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward
        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        var_cur = vars_cur[self.var_index]
        var_prev = vars_prev[self.var_index]

        if self.maintain and var_cur == var_prev or not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            self.cum_rew += self.rew
            reward += self.cum_rew
        else:
            self.cum_rew = 0
        return reward


class ProportionalVariableRewardWrapper(RewardWrapper):

    def __init__(self, env, scaler: float, var_index: int = 0, keep_lb: bool = False):
        super(ProportionalVariableRewardWrapper, self).__init__(env)
        self.scaler = scaler
        self.var_index = var_index
        self.keep_lb = keep_lb
        self.lower_bound = -np.inf

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            self.lower_bound = -np.inf
            return reward

        var_cur = self.game_variable_buffer[-1][self.var_index]
        var_prev = self.game_variable_buffer[-2][self.var_index]

        if not self.keep_lb or self.keep_lb and var_cur > self.lower_bound:
            reward = self.scaler * (var_cur - var_prev)
        self.lower_bound = max(var_cur, self.lower_bound) if self.keep_lb else 0
        return reward


class UserVariableRewardWrapper(RewardWrapper):

    def __init__(self, env, reward: float, game_var: GameVariable, decrease: bool = False,
                 update_callback: Callable = None):
        super(UserVariableRewardWrapper, self).__init__(env)
        self.rew = reward
        self.game_var = game_var
        self.decrease = decrease
        self.update_callback = update_callback

    def reward(self, reward):
        var_cur = self.game.get_game_variable(self.game_var)
        var_prev = self.get_and_update_user_var(self.game_var)

        if not self.decrease and var_cur > var_prev or self.decrease and var_cur < var_prev:
            reward += self.rew
        return reward


class MovementRewardWrapper(RewardWrapper):
    def __init__(self, env, scaler: float):
        super(MovementRewardWrapper, self).__init__(env)
        self.scaler = scaler

    def reward(self, reward):
        if len(self.distance_buffer) < 2:
            return reward
        distance = self.distance_buffer[-1]
        reward += distance * self.scaler  # Increase the reward for movement linearly
        return reward


class LocationVariableRewardWrapper(RewardWrapper):

    def __init__(self, env, x_index, y_index, x_start, y_start):
        super(LocationVariableRewardWrapper, self).__init__(env)
        self.x_index = x_index
        self.y_index = y_index
        self.x_start = x_start
        self.y_start = y_start

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward

        vars_cur = self.game_variable_buffer[-1]
        vars_prev = self.game_variable_buffer[-2]

        x_cur = vars_cur[self.x_index]
        y_cur = vars_cur[self.y_index]
        x_prev = vars_prev[self.x_index]
        y_prev = vars_prev[self.y_index]

        x_diff = max(0, abs(x_cur - self.x_start) - abs(x_prev - self.x_start))
        y_diff = max(0, abs(y_cur - self.y_start) - abs(y_prev - self.y_start))
        return self.reward_scaler_traversal * (x_diff + y_diff)


class PlatformReachedRewardWrapper(RewardWrapper):

    def __init__(self, env, reward: float, z_var_index: int = 0):
        super(PlatformReachedRewardWrapper, self).__init__(env)
        self.z_var_index = z_var_index
        self.rew = reward

    def reward(self, reward):
        if len(self.game_variable_buffer) < 2:
            return reward
        vars_cur = self.game_variable_buffer[-1]

        height_cur = vars_cur[self.z_var_index]
        heights_prev = [game_vars[self.z_var_index] for game_vars in self.game_variable_buffer]

        # Check whether the agent was on lava in the last n frames and is now on a platform
        if height_cur > max(heights_prev[:-1]):
            reward += self.rew
        return reward


class GoalRewardWrapper(RewardWrapper):

    def __init__(self, env, reward: float, goal: float, var_index: int = 0):
        super(GoalRewardWrapper, self).__init__(env)
        self.rew = reward
        self.goal = goal
        self.var_index = var_index

    def reward(self, reward):
        vars_cur = self.game_variable_buffer[-1]
        var_cur = vars_cur[self.var_index]

        if var_cur > self.goal:
            reward += self.rew
        return reward
