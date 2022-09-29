import os
from argparse import Namespace
from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Any, List

import gym
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution, Button, GameVariable

from coom.env.scenario.common import CommonEnv


class DoomEnv(CommonEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks: int):
        super().__init__()
        self.env_name = env
        self.id = task_id
        self.scenario = self.__module__.split('.')[-1]
        self.n_tasks = num_tasks
        self.frame_skip = args.frame_skip

        # Recording
        self.metadata['render.modes'] = 'rgb_array'
        self.record_every = args.record_every

        # Determine the directory of the doom scenario
        scenario_dir = f'{Path(__file__).parent.parent.resolve()}/maps/{self.scenario}'

        # Initialize the Doom game instance
        self.game = vzd.DoomGame()
        self.game.load_config(f"{scenario_dir}/conf.cfg")
        self.game.set_doom_scenario_path(f"{scenario_dir}/{env}.wad")
        self.game.set_window_visible(args.render)
        if args.render:  # Use a higher resolution for watching gameplay
            self.game.set_screen_resolution(ScreenResolution.RES_1600X1200)
        if args.acceleration:  # Add SPEED action to the available in-game actions
            actions = self.game.get_available_buttons()
            actions.append(Button.SPEED)
            self.game.set_available_buttons(actions)
        self.game.init()

        # Define the observation space
        self.game_res = (self.game.get_screen_height(), self.game.get_screen_width(), 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.game_res, dtype=np.uint8)

        # Define the action space
        self.acceleration = args.acceleration
        self.available_actions = self.get_available_actions()
        self.action_num = len(self.available_actions)
        self.action_space = gym.spaces.Discrete(self.action_num)

        # Initialize the user variable dictionary
        self.user_variables = {var: 0.0 for var in self.user_vars}

        # Initialize the game variable queue
        self.game_variable_buffer = deque(maxlen=args.variable_queue_len)
        for _ in range(args.variable_queue_len):
            self.game_variable_buffer.append(self.game.get_state().game_variables)

        # Register the gym environment specification
        self.spec = gym.envs.registration.EnvSpec(f"{self.name}-v0")

    @property
    def task(self) -> str:
        return self.env_name

    @property
    def name(self) -> str:
        return f'{self.scenario}-{self.env_name}'

    @property
    def task_id(self):
        return self.id

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

    def reset(self) -> np.ndarray:
        self.game.new_episode()
        self.clear_episode_statistics()
        observation = self.game.get_state().screen_buffer
        observation = np.transpose(observation, [1, 2, 0])
        return observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = self.available_actions[action]
        self.game.set_action(action)
        self.game.advance_action(self.frame_skip)

        state = self.game.get_state()
        reward = 0.0
        done = self.game.is_player_dead() or self.game.is_episode_finished() or not state
        info = {}

        observation = np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.float32(np.zeros(self.game_res))
        if not done:
            self.game_variable_buffer.append(state.game_variables)

        self.store_statistics(self.game_variable_buffer)
        return observation, reward, done, info

    def get_available_actions(self):
        actions = []
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        m_forward = [[0.0], [1.0]]
        execute = [[0.0], [1.0]]
        speed = [[0.0], [1.0]]
        for t in t_left_right:
            for m in m_forward:
                for e in execute:
                    if self.acceleration and m == [1.0]:
                        for s in speed:
                            actions.append(t + m + e + s)
                    else:
                        actions.append(t + m + e)
        return actions

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        metrics = self.extra_statistics(mode)
        ratio = (self.get_success() - self.performance_lower_bound) / self.performance_upper_bound
        metrics[f'{mode}/success'] = np.clip(ratio, 0.0, 1.0)
        return metrics

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {}

    def store_statistics(self, game_vars: deque) -> None:
        pass

    def get_success(self) -> float:
        raise NotImplementedError

    def reward_wrappers(self) -> List[gym.RewardWrapper]:
        raise NotImplementedError

    def get_and_update_user_var(self, game_var: GameVariable) -> int:
        prev_var = self.user_variables[game_var]
        self.user_variables[game_var] = self.game.get_game_variable(game_var)
        return prev_var

    def render(self, mode="rgb_array"):
        state = self.game.get_state()
        return np.transpose(state.screen_buffer, [1, 2, 0]) if state else np.uint8(np.zeros(self.game_res))

    def video_schedule(self, episode_id):
        return not episode_id % self.record_every

    def clear_episode_statistics(self) -> None:
        self.user_variables.fromkeys(self.user_variables, 0.0)

    def close(self):
        self.game.close()

    def get_active_env(self):
        return self
