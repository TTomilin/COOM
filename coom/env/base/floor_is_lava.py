from argparse import Namespace
from collections import deque
from typing import List

from coom.env.base.scenario import DoomEnv
from coom.env.utils.wrappers import WrapperHolder, ConstantRewardWrapper, GameVariableRewardWrapper


class FloorIsLava(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1, reward_frame_survived=0.01):
        super().__init__(args, env, task_id, num_tasks)
        self.penalty_health_loss = args.penalty_health_loss
        self.reward_frame_survived = reward_frame_survived
        self.frames_survived = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_health_loss, 0, True, True),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 2500.0  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 750.0  # Frames until the lava scorches the player

    def clear_episode_statistics(self) -> None:
        self.frames_survived = 0
