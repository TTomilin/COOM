import numpy as np
import time
from argparse import ArgumentParser, Namespace

from cl.utils.logx import Logger
from coom.envs import ContinualLearningEnv
from coom.utils.enums import Sequence


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--sequences', type=str, nargs='+', default=[Sequence.CO8, Sequence.CD8, Sequence.COC],
                        help='Sequences to run')
    parser.add_argument('--n_episodes', type=int, default=1, help='Number of episodes to run per task')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum number of steps per task')
    parser.add_argument('--render', type=bool, default=True, help='Whether to render the environment')
    parser.add_argument('--render_sleep', type=float, default=0.0, help='Time to sleep between rendering')
    return parser.parse_args()


def run(args: Namespace) -> None:
    doom_kwargs = dict(render=args.render, seed=np.random.randint(0, 2**32 - 1))
    logger = Logger('test_logs', config=vars(args), group_id='test')

    for sequence in args.sequences:
        print('\nRunning sequence:', sequence.name)
        cl_env = ContinualLearningEnv(logger, sequence, doom_kwargs=doom_kwargs)
        for env in cl_env.tasks:
            env.reset()
            steps = 0
            rewards = []
            for step in range(args.max_steps):
                state, reward, done, _, info = env.step(env.action_space.sample())
                steps += 1
                rewards.append(reward)
                if args.render:
                    env.render()
                    time.sleep(args.render_sleep)
                if done:
                    break
            print(f"Task {env.name} reward: {sum(rewards)}, steps: {steps}")
            env.close()


if __name__ == '__main__':
    run(get_args())
