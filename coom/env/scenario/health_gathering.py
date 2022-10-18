from argparse import Namespace
from collections import deque
from typing import List, Dict

from coom.env.scenario.scenario import DoomEnv
from coom.env.utils.wrappers import WrapperHolder, ConstantRewardWrapper, GameVariableRewardWrapper


class HealthGathering(DoomEnv):
    """
    In this scenario, the agent is trapped in a room with a surface, which slowly but constantly decreases the agent’s
    health. Health granting items continually spawn in random locations at specified time intervals. The default health
    item heals grants 25 hit points. Some environments contain poison vials, which inflict damage to the agent instead
    of providing health. The objective is to survive. The agent should identify the health granting items and navigate
    around the map to collect them quickly enough to avoid running out of health. The agent can turn left and right,
    and move forward. A small reward is granted for every frame the agent manages to survive.
    """

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1, reward_frame_survived=0.01):
        super().__init__(args, env, task_id, num_tasks)
        self.reward_item_acquired = args.reward_item_acquired
        self.reward_frame_survived = reward_frame_survived
        self.frames_survived = 0
        self.kits_obtained = 0

    def store_statistics(self, game_var_buf: deque) -> None:
        self.frames_survived += 1

        current_vars = game_var_buf[-1]
        previous_vars = game_var_buf[-2]
        if current_vars[0] > previous_vars[0]:
            self.kits_obtained += 1

    def get_success(self) -> float:
        return self.frames_survived * self.frame_skip

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(GameVariableRewardWrapper, self.reward_item_acquired, 0),
        ]

    @property
    def performance_upper_bound(self) -> float:
        return 2500.0  # Scenario length

    @property
    def performance_lower_bound(self) -> float:
        return 800.0  # Frames until the acid eats the player

    def extra_statistics(self, mode: str = '') -> Dict[str, float]:
        return {f'{mode}/kits_obtained': self.kits_obtained}

    def clear_episode_statistics(self) -> None:
        super().clear_episode_statistics()
        self.frames_survived = 0
        self.kits_obtained = 0
