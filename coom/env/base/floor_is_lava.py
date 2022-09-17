from argparse import Namespace
from typing import List

from coom.env.base.scenario import DoomEnv
from coom.env.utils.wrappers import WrapperHolder, ConstantRewardWrapper, GameVariableRewardWrapper


class FloorIsLava(DoomEnv):

    def __init__(self, args: Namespace, env: str, task_id: int, num_tasks=1, reward_frame_survived=0.01):
        self.reward_frame_survived = reward_frame_survived
        self.penalty_health_loss = args.penalty_health_loss
        super().__init__(args, env, task_id, num_tasks)

    def reward_wrappers(self) -> List[WrapperHolder]:
        return [
            WrapperHolder(ConstantRewardWrapper, self.reward_frame_survived),
            WrapperHolder(GameVariableRewardWrapper, self.penalty_health_loss, 0, True, True),
        ]
