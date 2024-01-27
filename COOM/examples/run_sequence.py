import argparse

from COOM.env.continual import ContinualLearningEnv
from COOM.utils.config import Sequence


def main(args: argparse.Namespace):
    sequence = Sequence[args.sequence.upper()]
    cl_env = ContinualLearningEnv(sequence)
    for env in cl_env.tasks:
        env.reset()
        total_reward = 0
        success = 0
        for steps in range(100):
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            success += env.get_success()
            env.render()
            if done:
                break
        print(f"Task {env.task_id}-{env.name} finished. Reward: {total_reward:.2f}. Success: {success / steps:.2f}")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Doom")
    parser.add_argument("--sequence", type=str, default='CO8',
                        choices=['CD4', 'CD8', 'CD16', 'CO4', 'CO8', 'CO16', 'COC', 'MIXED'],
                        help="Name of the continual learning sequence")
    main(parser.parse_args())
