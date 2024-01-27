import cv2
import gymnasium
import numpy as np
from gymnasium.spaces import Box
from typing import Tuple, Dict, Any

from COOM.env.scenario import DoomEnv
from COOM.utils.utils import combine_frames
from COOM.utils.config import Augmentation


class Rescale(gymnasium.Wrapper):
    """Rescale the observation space to [-1, 1]."""

    def __init__(self, env):
        gymnasium.Wrapper.__init__(self, env)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset()
        return state / 255. * 2 - 1, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        return state / 255. * 2 - 1, reward, done, truncated, info


class Resize(gymnasium.Wrapper):
    """Resize the observation space."""

    def __init__(self, env, height=84, width=84):
        gymnasium.Wrapper.__init__(self, env)
        assert height > 0 and width > 0, f"Invalid shape: {height}x{width}"
        self.shape = (height, width)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        state, info = self.env.reset()
        return cv2.resize(state, self.shape), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        state, reward, done, truncated, info = self.env.step(action)
        return cv2.resize(state, self.shape), reward, done, truncated, info


class RGBStack(gymnasium.Wrapper):
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


class Augment(gymnasium.Wrapper):
    """Augment the visual observation"""

    def __init__(self, env: DoomEnv, augmentation: str):
        super(Augment, self).__init__(env)
        assert augmentation.upper() in Augmentation.__members__, f"Unknown augmentation: {augmentation}"
        self.augmentation = Augmentation[augmentation.upper()].value

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset()
        obs = self.augmentation(obs)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, truncated, info = self.env.step(action)
        obs = self.augmentation(obs)
        return obs, reward, done, truncated, info
