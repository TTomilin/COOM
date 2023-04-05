import argparse
from collections import deque

import cv2
import gym
import numpy as np
import time
import vizdoom as vzd
from pathlib import Path
from typing import Dict, Tuple, Any, List
from vizdoom import ScreenResolution, GameVariable

from coom.env.scenario.common import CommonEnv
from coom.env.utils.utils import get_screen_resolution, default_action_space


class DoomEnv(CommonEnv):

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser):
        def arg(*args, **kwargs):
            try:
                parser.add_argument(*args, **kwargs)
            except argparse.ArgumentError:
                pass  # Argument already exists

        arg('--render', default=False, action='store_true', help='Render the environment')
        arg('--render_mode', type=str, default='rgb_array', help='Mode of rendering')
        arg('--render_sleep', type=float, default=0.0, help='Sleep time between frames when rendering')
        arg('--variable_queue_length', type=int, default=5, help='Number of game variables to remember')
        arg('--frame_skip', type=int, default=4, help='Number of frames to skip')
        arg('--resolution', type=str, default=None, choices=['800x600', '640x480', '320x240', '160x120'],
            help='Screen resolution of the game')

    def __init__(self,
                 env: str = 'default',
                 task_idx: int = 0,
                 num_tasks: int = 1,
                 frame_skip: int = 4,
                 record_every: int = 100,
                 seed: int = 0,
                 render: bool = False,
                 render_mode: str = 'rgb_array',
                 render_sleep: float = 0.0,
                 test_only: bool = False,
                 resolution: str = None,
                 variable_queue_length: int = 5):
        super().__init__()
        self.env_name = env
        self.task_idx = task_idx
        self.scenario = self.__module__.split('.')[-1]
        self.n_tasks = num_tasks
        self.frame_skip = frame_skip

        # Recording
        self.metadata['render.modes'] = 'rgb_array'
        self.record_every = record_every
        self.viewer = None

        # Determine the directory of the doom scenario
        scenario_dir = f'{Path(__file__).parent.parent.resolve()}/maps/{self.scenario}'

        # Initialize the Doom game instance
        self.game = vzd.DoomGame()
        self.game.load_config(f"{scenario_dir}/conf.cfg")
        self.game.set_doom_scenario_path(f"{scenario_dir}/{env}.wad")
        self.game.set_seed(seed)
        self.render_mode = render_mode
        self.render_sleep = render_sleep
        self.render_enabled = render
        if render or test_only:  # Use a higher resolution for watching gameplay
            self.game.set_screen_resolution(ScreenResolution.RES_1600X1200)
            self.frame_skip = 1
        elif resolution:  # Use a particular predefined resolution
            self.game.set_screen_resolution(get_screen_resolution(resolution))
        self.game.init()

        # Define the observation space
        self.game_res = (self.game.get_screen_height(), self.game.get_screen_width(), 3)
        self._observation_space = gym.spaces.Box(low=0, high=255, shape=self.game_res, dtype=np.uint8)

        # Define the action space
        self.available_actions = default_action_space()
        self._action_space = gym.spaces.Discrete(len(self.available_actions))

        # Initialize the user variable dictionary
        self.user_variables = {var: 0.0 for var in self.user_vars}

        # Initialize the game variable queue
        self.game_variable_buffer = deque(maxlen=variable_queue_length)

    @property
    def task(self) -> str:
        return self.env_name

    @property
    def name(self) -> str:
        return f'{self.scenario}-{self.env_name}'

    @property
    def task_id(self):
        return self.task_idx

    @property
    def num_tasks(self) -> int:
        return self.n_tasks

    @property
    def user_vars(self) -> List[GameVariable]:
        return []

    @property
    def performance_upper_bound(self) -> float:
        raise NotImplementedError

    @property
    def performance_lower_bound(self) -> float:
        raise NotImplementedError

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._action_space

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            self.game.new_episode()
        except vzd.ViZDoomIsNotRunningException:
            print('ViZDoom is not running. Restarting...')
            self.game.init()
            self.game.new_episode()
        self.clear_episode_statistics()
        state = self.game.get_state().screen_buffer
        state = np.transpose(state, [1, 2, 0])
        return state, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = self.available_actions[action]
        self.game.set_action(action)
        self.game.advance_action(self.frame_skip)

        state = self.game.get_state()
        reward = 0.0
        done = self.game.is_player_dead() or self.game.is_episode_finished() or not state
        truncated = False
        info = {}

        observation = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.float32(np.zeros(self.game_res))
        if not done:
            self.game_variable_buffer.append(state.game_variables)
        if self.render_enabled:
            self.render()
            time.sleep(self.render_sleep)

        self.store_statistics(self.game_variable_buffer)
        return observation, reward, done, truncated, info

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        metrics = self.extra_statistics(mode)
        ratio = (self.get_success() - self.performance_lower_bound) / (
                self.performance_upper_bound - self.performance_lower_bound)
        metrics[f'{mode}/success'] = np.clip(ratio, 0.0, 1.0)
        return metrics

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {}

    def store_statistics(self, game_vars: deque) -> None:
        pass

    def get_success(self) -> float:
        raise NotImplementedError

    def reward_wrappers_dense(self) -> List[gym.RewardWrapper]:
        raise NotImplementedError

    def reward_wrappers_sparse(self) -> List[gym.RewardWrapper]:
        raise NotImplementedError

    def get_and_update_user_var(self, game_var: GameVariable) -> int:
        prev_var = self.user_variables[game_var]
        self.user_variables[game_var] = self.game.get_game_variable(game_var)
        return prev_var

    def render(self, mode="rgb_array"):
        state = self.game.get_state()
        img = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.uint8(np.zeros(self.game_res))
        if mode == 'human':
            if not self.render_enabled:
                return [img]
            try:
                h, w = img.shape[:2]
                render_w = 1280

                if w < render_w:
                    render_h = int(render_w * h / w)
                    img = cv2.resize(img, (render_w, render_h))

                # Render the image to the screen with swapped red and blue channels
                cv2.imshow('DOOM', img[:, :, [2, 1, 0]])
                cv2.waitKey(1)
            except Exception as e:
                print('Screen rendering unsuccessful', e)
                return np.zeros(img.shape)
        return [img]

    def video_schedule(self, episode_id):
        return not episode_id % self.record_every

    def clear_episode_statistics(self) -> None:
        self.user_variables.fromkeys(self.user_variables, 0.0)
        self.game_variable_buffer.clear()

    def close(self):
        self.game.close()

    def get_active_env(self):
        return self
