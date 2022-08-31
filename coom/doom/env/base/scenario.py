from argparse import Namespace
from collections import deque
from typing import Dict, Tuple, Any, List

import gym
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution, Button


class DoomEnv(gym.Env):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks: int):
        super().__init__()
        self.task = task
        self.name = task
        self.task_id = task_id
        self.num_tasks = num_tasks
        self.frame_skip = args.frame_skip

        # Create the Doom game instance
        self.game = vzd.DoomGame()
        self.game.load_config(args.cfg_path)
        self.game.set_doom_scenario_path(f"{args.experiment_dir}/coom/doom/maps/{args.scenario}/{task}.wad")  # TODO remove hard coded path
        self.game.set_window_visible(args.render)
        if args.watch or args.render:  # Use a higher resolution for watching gameplay
            self.game.set_screen_resolution(ScreenResolution.RES_1600X1200)
        if args.add_speed:  # Add SPEED action to the available in-game actions
            actions = self.game.get_available_buttons()
            actions.append(Button.SPEED)
            self.game.set_available_buttons(actions)
        self.game.init()

        # Define the observation space
        self.game_res = (self.game.get_screen_height(), self.game.get_screen_width())
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.game_res, dtype=np.uint8)

        # Define the action space
        self.available_actions = self.get_available_actions()
        self.action_num = len(self.available_actions)
        self.action_space = gym.spaces.Discrete(self.action_num)

        # Initialize the game variable queue
        self.game_variable_buffer = deque(maxlen=args.variable_queue_len)
        for _ in range(args.variable_queue_len):
            self.game_variable_buffer.append(self.game.get_state().game_variables)

        self.spec = gym.envs.registration.EnvSpec(f"{task}-v0")

    def reset(self) -> np.ndarray:
        self.game.new_episode()
        self.clear_episode_statistics()
        return self.game.get_state().screen_buffer

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = self.available_actions[action]
        self.game.set_action(action)
        self.game.advance_action(self.frame_skip)

        state = self.game.get_state()
        reward = self.calc_reward()
        done = self.game.is_player_dead() or self.game.is_episode_finished() or not state
        info = self.get_statistics()

        observation = state.screen_buffer if state else np.zeros(self.game_res)
        if not done:
            self.game_variable_buffer.append(state.game_variables)
        return observation, reward, done, info

    def calc_reward(self) -> float:
        raise NotImplementedError

    def get_available_actions(self) -> List[List[float]]:
        raise NotImplementedError

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        return {}

    def render(self, mode="human"):
        pass

    def clear_episode_statistics(self) -> None:
        pass

    def close(self):
        self.game.close()

    def get_active_env(self):
        return self
