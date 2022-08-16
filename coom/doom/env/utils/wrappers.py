from typing import Dict, Any, Tuple

import cv2
import gym
import numpy as np
from gym.spaces import Box


class RescaleWrapper(gym.Wrapper):
    """Scales the observation space to [-1, 1]."""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self) -> np.ndarray:
        return self.env.reset() / 255. * 2 - 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, reward, done, info = self.env.step(action)
        return state / 255. * 2 - 1, reward, done, info


class NormalizeWrapper(gym.Wrapper):
    """Normalize the observation space."""

    def __init__(self, env, eps=1e-6):
        gym.Wrapper.__init__(self, env)
        self.eps = eps

    def reset(self) -> np.ndarray:
        return self.env.reset() / 255. * 2 - 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, reward, done, info = self.env.step(action)
        mean = self.states.mean()
        std = self.states.std() + self.eps
        state = (state - mean) / std
        return state, reward, done, info


class ResizeWrapper(gym.Wrapper):
    """Rescale the observation space."""

    def __init__(self, env, height=84, width=84):
        gym.Wrapper.__init__(self, env)
        self.shape = (height, width)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def reset(self) -> np.ndarray:
        state = self.env.reset()
        return cv2.resize(state, self.shape)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, reward, done, info = self.env.step(action)
        return cv2.resize(state, self.shape), reward, done, info
