from typing import Dict, List

import gym


class CommonEnv(gym.Env):

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self, mode="human"):
        raise NotImplementedError

    @property
    def task(self) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def task_id(self) -> int:
        raise NotImplementedError

    @property
    def num_tasks(self) -> int:
        raise NotImplementedError

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        raise NotImplementedError

    def clear_episode_statistics(self) -> None:
        raise NotImplementedError

    def get_shared_action_space(self) -> List[List[float]]:
        actions = []
        m_forward = [[0.0], [1.0]]
        t_left_right = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        attack = [[0.0], [1.0]]
        use = [[0.0], [1.0]]
        speed = [[0.0], [1.0]]

        for t in t_left_right:
            for m in m_forward:
                for a in attack:
                    for u in use:
                        if self.add_speed and m == [1.0]:
                            for s in speed:
                                actions.append(t + m + a + u + s)
                        else:
                            actions.append(t + m + a + u)
        return actions