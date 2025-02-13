from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Dict


class Saute(gym.Wrapper):
    """Saute Adapter for Doom.

    Saute is a safe RL algorithm that uses state augmentation to ensure safety. The state
    augmentation is the concatenation of the original state and the safety state. The safety state
    is the safety budget minus the cost divided by the safety budget.

    .. note::
        - If the safety state is greater than 0, the reward is the original reward.
        - If the safety state is less than 0, the reward is the unsafe reward (always 0 or less than 0).

    References:
        - Title: Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation
        - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang,
            David Mguni, Jun Wang, Haitham Bou-Ammar.
        - URL: `Saute <https://arxiv.org/abs/2202.06558>`_

    Args:
        env (Env): The gymnasium environment being wrapped.
        saute_gamma (float): The discount factor for the safety budget calculation.
    """

    def __init__(self, env, saute_gamma: float):
        super().__init__(env)
        self.saute_gamma = saute_gamma
        self.safety_obs = 1
        self.episode_reward = 0
        self.safety_budget = self.safety_bound * (1 - saute_gamma ** self.timeout) / (1 - saute_gamma) / self.timeout

        obs_space = self.env.observation_space
        assert isinstance(obs_space, Box), 'Observation space must be Box'
        self.env.observation_space = Dict({
            'obs': Box(low=0, high=255, shape=obs_space.shape, dtype=np.uint8),
            'safety': Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        .. note::
            Additionally, the safety observation will be reset.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.safety_obs = 1
        self.episode_reward = 0
        obs = {
            'obs': obs,
            'safety:': self.safety_obs
        }
        return obs, info

    def step(self, action):
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The :meth:`_saute_step` will be called to update the safety observation. Then the reward
            will be updated by :meth:`_safety_reward`.

        Args:
            action: The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        orig_rew = reward
        self.episode_reward += reward
        cost = info.get('cost', 0.0)

        self._safety_step(cost)
        reward = self._safety_reward(reward)

        # autoreset the environment
        done = terminated or truncated
        self.safety_obs = self.safety_obs * (1 - float(done)) + float(done)

        # Update episode extra stats in info
        self.update_episode_stats(done, info, orig_rew)

        obs = {
            'obs': obs,
            'safety:': self.safety_obs
        }

        return obs, reward, terminated, truncated, info

    def _safety_step(self, cost: torch.Tensor) -> None:
        """Update the safety observation.

        Args:
            cost (float): The cost of the current step.
        """
        self.safety_obs -= cost / self.safety_budget
        self.safety_obs /= self.saute_gamma

    def _safety_reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Update the reward with the safety observation.

        .. note::
            If the safety observation is greater than 0, the reward will be the original reward.
            Otherwise, the reward will be the unsafe reward.

        Args:
            reward (float): The reward of the current step.

        Returns:
            The final reward determined by the safety observation.
        """
        return reward if self.safety_obs > 0 else self.unsafe_reward

    def update_episode_stats(self, done, info, orig_rew):
        key = 'episode_extra_stats'
        if key not in info:
            info[key] = {}
        info[key]['original_reward'] = orig_rew
        info[key]['safety_obs'] = self.safety_obs
        if done:
            info[key]['episode_reward'] = self.episode_reward
