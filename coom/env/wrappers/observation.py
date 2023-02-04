import cv2
import numpy as np
import tensorflow as tf
from gym.spaces import Box
from typing import Tuple, Dict, Any

import gym


class RescaleWrapper(gym.Wrapper):
    """Rescale the observation space to [-1, 1]."""

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset()
        return state / 255. * 2 - 1, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        return state / 255. * 2 - 1, reward, done, truncated, info


class NormalizeWrapper(gym.Wrapper):
    """Normalize the observation space."""

    def __init__(self, env, eps=1e-6):
        gym.Wrapper.__init__(self, env)
        self.eps = eps

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset()
        mean = self.states.mean()
        std = self.states.std() + self.eps
        state = (state - mean) / std
        return state / 255. * 2 - 1, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        mean = self.states.mean()
        std = self.states.std() + self.eps
        state = (state - mean) / std
        return state, reward, done, truncated, info


class ResizeWrapper(gym.Wrapper):
    """Resize the observation space."""

    def __init__(self, env, height=84, width=84):
        gym.Wrapper.__init__(self, env)
        self.shape = (height, width)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset()
        return cv2.resize(state, self.shape), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        return cv2.resize(state, self.shape), reward, done, truncated, info


class RGBStack(gym.Wrapper):
    """Combine the stacked frames with RGB colours. [n_stack, h, w, 3] -> [h, w, n_stack * 3]"""

    def __init__(self, env):
        super(RGBStack, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=(obs_shape[1], obs_shape[2], obs_shape[0] * obs_shape[3]), dtype=np.uint8
        )

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset()
        state = combine_frames(state)
        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        state = combine_frames(state)
        return state, reward, done, truncated, info


def combine_frames(obs):
    return tf.reshape(obs, [obs.shape[1], obs.shape[2], obs.shape[0] * obs.shape[3]])
