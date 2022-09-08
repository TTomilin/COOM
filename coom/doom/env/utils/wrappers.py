from typing import Dict, Any, Tuple

import tensorflow as tf

import cv2
import gym
import numpy as np
from gym.spaces import Box


class RescaleWrapper(gym.Wrapper):
    """Rescale the observation space to [-1, 1]."""

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
    """Resize the observation space."""

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


class RGBStack(gym.Wrapper):
    """Combine the stacked frames with RGB colours. [n_stack, h, w, 3] -> [h, w, n_stack * 3]"""

    def __init__(self, env):
        super(RGBStack, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            low=0, high=255, shape=(obs_shape[1], obs_shape[2], obs_shape[0] * obs_shape[3]), dtype=np.uint8
        )

    def reset(self) -> np.ndarray:
        observation = self.env.reset()
        observation = combine_frames(observation)
        return observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        observation, reward, done, info = self.env.step(action)
        observation = combine_frames(observation)
        return observation, reward, done, info


def combine_frames(obs):
    return tf.reshape(obs, [obs.shape[1], obs.shape[2], obs.shape[0] * obs.shape[3]])
