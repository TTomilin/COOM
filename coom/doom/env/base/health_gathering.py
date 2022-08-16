from argparse import Namespace

from coom.doom.env.base.scenario import DoomEnv


class HealthGathering(DoomEnv):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, frame_survived_reward=0.01):
        self.add_speed = args.add_speed
        self.reward_frame_survived = frame_survived_reward
        super().__init__(args, task, task_id, num_tasks)

    def get_available_actions(self):
        actions = []
        speed = [[0.0], [1.0]]
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        for m in m_forward:
            for t in t_left_right:
                if self.add_speed:
                    for s in speed:
                        actions.append(t + m + s)
                else:
                    actions.append(t + m)
        return actions

    def calc_reward(self) -> float:
        return self.reward_frame_survived
