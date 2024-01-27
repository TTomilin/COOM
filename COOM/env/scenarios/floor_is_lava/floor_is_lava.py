from collections import deque

import numpy as np
from typing import List, Dict

from COOM.env.scenario import DoomEnv
from COOM.utils.utils import distance_traversed
from COOM.wrappers.reward import WrapperHolder, GameVariableRewardWrapper, \
    PlatformReachedRewardWrapper, ConstantRewardWrapper, CumulativeVariableRewardWrapper


class FloorIsLava(DoomEnv):
    """
    In this scenario, the agent is located in a square-shaped room divided into 16x16 equal sized squares. The room is
    filled with lava, which inflicts 1 health point of damage if being stood upon. After every fixed time interval each
    square section of lava has a 20% chance of being changed to a platform, which no longer causes damage. The objective
    is to survive by minimizing the time spent standing in lava. The agent ought to identify the platforms and quickly
    navigate on top of them as soon as they appear to avoid running out of health. The agent can turn left and right,
    move forward, and accelerate. A small reward is granted for every frame the agent manages to survive with sparse
    rewards. With auxiliary rewards, the agent is rewarded for stepping onto a platform whilst having previously stood
    in lava, and penalized for standing in lava.
    """

    def __init__(self,
                 doom_kwargs: Dict[str, any],
                 reward_on_platform: float = 0.1,
                 reward_scaler_traversal: float = 1e-3,
                 reward_platform_reached: float = 1.0,
                 reward_frame_survived: float = 0.01,
                 penalty_lava: float = -0.1):
        super().__init__(**doom_kwargs)
        self.penalty_lava = penalty_lava
        self.reward_on_platform = reward_on_platform
        self.reward_frame_survived = reward_frame_survived
        self.reward_scaler_traversal = reward_scaler_traversal
        self.reward_platform_reached = reward_platform_reached
        self.distance_buffer = []
        self.frames_survived = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) > 1:
            distance = distance_traversed(game_var_buf, 1, 2)
            self.distance_buffer.append(distance)

    def get_success_metric(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers_dense(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(PlatformReachedRewardWrapper, reward=self.reward_platform_reached, z_var_index=3),
            WrapperHolder(CumulativeVariableRewardWrapper, reward=self.reward_on_platform, var_index=0, maintain=True),
            WrapperHolder(GameVariableRewardWrapper, reward=self.penalty_lava, var_index=0, decrease=True)
        ]

    def reward_wrappers_sparse(self) -> List[WrapperHolder]:
        return [WrapperHolder(ConstantRewardWrapper, reward=self.reward_frame_survived)]

    @property
    def performance_upper_bound(self) -> float:
        return 2500  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 700  # Frames until the lava scorches the player

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
