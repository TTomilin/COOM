from collections import deque

import numpy as np
from argparse import Namespace
from typing import List, Dict

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.wrappers import WrapperHolder, GameVariableRewardWrapper, \
    MovementRewardWrapper


class FloorIsLava(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        super().__init__(args, env, task_id, num_tasks)
        self.penalty_health_loss = args.penalty_health_loss
        self.reward_frame_survived = args.reward_frame_survived
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.distance_buffer = []
        self.frames_survived = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            # WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_health_loss, 0, True),
            WrapperHolder(MovementRewardWrapper, self.reward_scaler_traversal),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 2500  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 800  # Frames until the lava scorches the player

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
