from collections import deque

import numpy as np
from typing import List, Dict
from vizdoom import GameVariable

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.wrappers.reward import MovementRewardWrapper, WrapperHolder, ConstantRewardWrapper, \
    UserVariableRewardWrapper


class RaiseTheRoof(DoomEnv):
    """
    In this scenario, the agent is randomly spawned in one of 20 possible locations within a maze-like environment. The
    room has a very high ceiling at the beginning of an episode, which, however, starts to be slowly but constantly
    lowered. At specific locations of the area, there are switches on the walls that can be pressed to raise the ceiling
    back up a bit. After a switch is pressed, it disappears and can no longer be used. If the switches are not pressed
    with a high enough frequency, the ceiling will eventually crush the agent, which terminates the episode. The goal of
    the agent is thus to locate and press the switches to keep the ceiling high before the episodes timeouts. The agent
    can move forward, turn left and right, and activate a switch. In the sparse reward case, the agent only acquires a
    reward for surviving a frame. With dense rewards, the agent is rewarded for movement, pressing a switch, and
    surviving for a frames.
    """

    def __init__(self,
                 doom_kwargs: Dict[str, any],
                 reward_scaler_traversal: float = 0.001,
                 reward_frame_survived: float = 0.01,
                 reward_switch_pressed: float = 15.0):
        super().__init__(**doom_kwargs)
        self.reward_scaler_traversal = reward_scaler_traversal
        self.reward_frame_survived = reward_frame_survived
        self.reward_switch_pressed = reward_switch_pressed
        self.distance_buffer = []
        self.frames_survived = 0

    @property
    def user_vars(self) -> List[GameVariable]:
        return [GameVariable.USER2]

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) > 1:
            distance = distance_traversed(game_var_buf, 0, 1)
            self.distance_buffer.append(distance)

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers_dense(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, reward=self.reward_frame_survived),
            WrapperHolder(UserVariableRewardWrapper, reward=self.reward_switch_pressed, game_var=GameVariable.USER2),
            WrapperHolder(MovementRewardWrapper, scaler=self.reward_scaler_traversal),
        ]

    def reward_wrappers_sparse(self) -> List[WrapperHolder]:
        return [WrapperHolder(ConstantRewardWrapper, reward=self.reward_frame_survived)]

    @property
    def performance_upper_bound(self) -> float:
        return 2500.0  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 640.0  # Frames until getting crushed

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/switches_pressed': self.user_variables[GameVariable.USER2]}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
