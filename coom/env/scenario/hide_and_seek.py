from collections import deque

import numpy as np
from argparse import Namespace
from typing import List, Dict

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.utils import distance_traversed
from coom.env.utils.wrappers import MovementRewardWrapper, WrapperHolder, GameVariableRewardWrapper


class HideAndSeek(DoomEnv):
    """
    In this scenario, the agent is randomly spawned in one of 20 possible locations within a maze-like environment.
    5 enemies are spawned at random locations at the beginning of an episode. The enemies can only inflict damage at
    close distance and thus constantly attempt to move closer to the agent. Health kits granting 25 hit points
    continually spawn in random locations at specified time intervals. The objective of the agent is to survive by
    hiding from the enemies. The agent should identify the enemies and attempt to constantly move away from them while
    collecting the health kits when at low health. The agent can turn left and right, move forward, and run. In the
    sparse reward case, a small reward is granted for every frame the agent manages to survive. With auxiliary rewards,
    the agent is rewarded for movement and each health item collected, and penalized for having damage inflicted by
    enemies.
    """

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_scaler_traversal = args.reward_scaler_traversal
        self.reward_health = args.reward_health_has
        self.penalty_health_loss = args.penalty_health_has
        self.distance_buffer = []
        self.frames_survived = 0
        self.kits_obtained = 0
        self.hits_taken = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1
        if len(game_var_buf) < 2:
            return

        distance = distance_traversed(game_var_buf, 1, 2)
        self.distance_buffer.append(distance)

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] < previous_vars[0]:
            self.hits_taken += 1
        elif current_vars[0] > previous_vars[0]:
            self.kits_obtained += 1

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(GameVariableRewardWrapper, self.reward_health, 0),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_health_loss, 0, True),
            WrapperHolder(MovementRewardWrapper, self.reward_scaler_traversal),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 2500  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 900  # Frames until getting slaughtered by the monsters when taking random actions

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        if not self.game_variable_buffer:
            return {}
        variables = self.game_variable_buffer[-1]
        return {f'{mode}/health': variables[0],
                f'{mode}/movement': np.mean(self.distance_buffer).round(3),
                f'{mode}/hits_taken': self.hits_taken,
                f'{mode}/kits_obtained': self.kits_obtained}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.distance_buffer.clear()
        self.frames_survived = 0
        self.kits_obtained = 0
        self.hits_taken = 0
