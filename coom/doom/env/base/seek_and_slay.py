from argparse import Namespace

from coom.doom.env.base.scenario import DoomEnv


class SeekAndSlay(DoomEnv):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1, kill_reward=1.0):
        self.add_speed = args.add_speed
        self.kill_reward = kill_reward
        super().__init__(args, task, task_id, num_tasks)

    def get_available_actions(self):
        actions = []
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        attack = [[0.0], [1.0]]
        speed = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward:
                for a in attack:
                    if self.add_speed:
                        for s in speed:
                            actions.append(t + m + a + s)
                    else:
                        actions.append(t + m + a)
        return actions

    def calc_reward(self) -> float:
        reward = 0.0
        current_vars = self.game_variable_buffer[-1]
        previous_vars = self.game_variable_buffer[-2]

        if current_vars[1] > previous_vars[1]:
            reward += self.kill_reward  # Elimination of enemy

        return reward
