import gymnasium as gym
from vizdoom import GameVariable

STARTING_HEALTH = 100
HEALTH_COST_SCALER = 0.04


class DoomDetonatorsDilemmaCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._prev_cost = 0
        self._episode_reward = 0
        self._total_health_cost = 0
        self._prev_health = STARTING_HEALTH

    def reset(self, **kwargs):
        self._prev_cost = 0
        self._episode_reward = 0
        self._total_health_cost = 0
        self._prev_health = STARTING_HEALTH
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        cost = self.game.get_game_variable(GameVariable.USER1)
        health = info['HEALTH']
        health_cost = (self._prev_health - health) * HEALTH_COST_SCALER
        self._total_health_cost += health_cost
        cost_this_step = cost - self._prev_cost + health_cost

        self._prev_cost = cost
        self._prev_health = health
        self._episode_reward += reward

        info['cost'] = cost_this_step
        info['true_objective'] = reward
        info["episode_extra_stats"] = {
            'cost': cost + self._total_health_cost,
            'health_cost': health_cost,
            'ammo': info['AMMO2'],
            'kills': info['KILLCOUNT'],
            'episode_reward': self._episode_reward,
        }

        return observation, reward, terminated, truncated, info
