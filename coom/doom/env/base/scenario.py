import os
from argparse import Namespace
from collections import deque
from typing import Dict

import cv2
import gym
import numpy as np
import vizdoom as vzd
from vizdoom import ScreenResolution


class DoomEnv(gym.Env):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks: int):
        super().__init__()
        self.name = args.scenario
        self.task_id = task_id
        self.num_tasks = num_tasks
        self.save_lmp = args.save_lmp
        if args.save_lmp:
            os.makedirs("lmps", exist_ok=True)
        self.res = args.res
        self.skip = args.frames_stack
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=args.res, dtype=np.float32
        )
        # TODO remove hard coded path
        wad_path = f"{args.experiment_dir}/coom/doom/maps/{args.scenario}/{task}.wad"
        self.game = vzd.DoomGame()
        self.game.load_config(args.cfg_path)
        self.game.set_doom_scenario_path(wad_path)
        self.game.set_window_visible(args.render)
        if args.watch:
            self.game.set_screen_resolution(ScreenResolution.RES_1600X1200)
        self.game.init()

        # Initialize and fill game variable queue
        self.game_variable_buffer = deque(maxlen=args.variable_queue_len)
        for _ in range(args.variable_queue_len):
            self.game_variable_buffer.append(self.game.get_state().game_variables)

        self.extra_statistics = ['kills', 'health', 'ammo', 'movement', 'kits_obtained', 'hits_taken']
        self.available_actions = self.get_available_actions()
        self.action_num = len(self.available_actions)
        self.action_space = gym.spaces.Discrete(self.action_num)
        self.spec = gym.envs.registration.EnvSpec("levdoom-v0")
        self.count = 0

    def get_obs(self):
        state = self.game.get_state()
        if state is None:
            return
        obs = state.screen_buffer
        self.obs_buffer[:-1] = self.obs_buffer[1:]
        self.obs_buffer[-1] = cv2.resize(obs, (self.res[-1], self.res[-2]))

    def reset(self):
        if self.save_lmp:
            self.game.new_episode(f"lmps/episode_{self.count}.lmp")
        else:
            self.game.new_episode()
        self.count += 1
        self.obs_buffer = np.zeros(self.res, dtype=np.uint8)
        self.get_obs()
        self.clear_episode_statistics()
        return self.obs_buffer

    def step(self, action):
        action = self.available_actions[action]
        self.game.set_action(action)
        self.game.advance_action(self.skip)
        self.get_obs()
        reward = self.calc_reward()
        state = self.game.get_state()
        done = self.game.is_player_dead() or not state or self.game.is_episode_finished()
        if not done:
            self.game_variable_buffer.append(state.game_variables)
        info = self.get_statistics()
        info['task_id'] = self.task_id
        info['num_tasks'] = self.num_tasks
        return self.obs_buffer, reward, done, info

    def calc_reward(self) -> float:
        raise NotImplementedError

    def get_available_actions(self):
        raise NotImplementedError

    def get_statistics(self) -> Dict[str, float]:
        return {}

    def render(self):
        pass

    def clear_episode_statistics(self):
        pass

    def close(self):
        self.game.close()
