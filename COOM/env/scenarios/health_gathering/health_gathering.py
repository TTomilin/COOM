import numpy as np
from collections import deque
from typing import List, Dict

from COOM.env.scenario import DoomEnv
from COOM.utils.utils import distance_traversed
from COOM.wrappers.reward import WrapperHolder, ConstantRewardWrapper, GameVariableRewardWrapper


class HealthGathering(DoomEnv):
    """
    In this scenario, the agent is trapped in a room with a surface, which slowly but constantly decreases the agent’s
    health. Health granting items continually spawn in random locations at specified time intervals. The default health
    item heals grants 25 hit points. Some environments contain poison vials, which inflict damage to the agent instead
    of providing health. The objective is to survive. The agent should identify the health granting items and navigate
    around the map to collect them quickly enough to avoid running out of health. The agent can turn left and right,
    and move forward. A small reward is granted for every frame the agent manages to survive.
    """

    def __init__(self,
                 doom_kwargs: Dict[str, any],
                 reward_health_hg: float = 15.0,
                 reward_frame_survived: float = 0.01,
                 penalty_health_hg: float = -0.01):
        super().__init__(**doom_kwargs)
        self.reward_frame_survived = reward_frame_survived
        self.reward_health_kit = reward_health_hg
        self.penalty_health_loss = penalty_health_hg
        self.distance_buffer = []
        self.frames_survived = 0
        self.kits_obtained = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return

        distance = distance_traversed(game_var_buf, 1, 2)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] > previous_vars[0]:
            self.kits_obtained += 1

    def get_success_metric(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers_dense(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, reward=self.penalty_health_loss),
            WrapperHolder(GameVariableRewardWrapper, reward=self.reward_health_kit, var_index=0),
        ]

    def reward_wrappers_sparse(self) -> List[WrapperHolder]:
        return [WrapperHolder(ConstantRewardWrapper, reward=self.reward_frame_survived)]

    @property
    def performance_upper_bound(self) -> float:
        return 2500.0  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 800.0  # Frames until the acid eats the player

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/kits_obtained': self.kits_obtained,
                f'{mode}/movement': np.mean(self.distance_buffer).round(3)}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
        self.kits_obtained = 0
