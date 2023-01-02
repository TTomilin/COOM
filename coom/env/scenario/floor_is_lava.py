from collections import deque

import numpy as np
from argparse import Namespace
from typing import List, Dict

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import WrapperHolder, GameVariableRewardWrapper, \
    MovementRewardWrapper, PlatformReachedRewardWrapper


class FloorIsLava(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        super().__init__(args, env, task_id, num_tasks)
        self.penalty_lava = args.penalty_lava
        self.reward_frame_survived = args.reward_frame_survived
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_platform_reached = args.reward_platform_reached
        self.distance_buffer = []
        self.frames_survived = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) > 1:
            distance = distance_traversed(game_var_buf, 1, 2)
            self.distance_buffer.append(distance)


    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(PlatformReachedRewardWrapper, self.reward_platform_reached, 0),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_lava, 0, True)
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 3500  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 1600  # Frames until the lava scorches the player

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
