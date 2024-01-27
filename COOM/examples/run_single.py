import argparse

from COOM.env.builder import make_env
from COOM.utils.config import Scenario


def main(args: argparse.Namespace):
    scenario = Scenario[args.scenario.upper()]
    env = make_env(scenario, args.task)
    env.reset()
    total_reward = 0
    success = 0
    for steps in range(1000):
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
    parser.add_argument('--scenario', type=str, default='health_gathering',
                        choices=['health_gathering', 'run_and_gun', 'chainsaw', 'raise_the_roof', 'floor_is_lava',
                                 'hide_and_seek', 'arms_dealer', 'pitfall'])
    parser.add_argument("--task", type=str, default='hard',
                        help="Name of the environments in the scenario(s) to run")
    main(parser.parse_args())
