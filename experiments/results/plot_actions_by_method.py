import json
import os
from matplotlib import pyplot as plt

from experiments.results.common import *

TRANSLATIONS = {
    'packnet': 'PackNet',
    'mas': 'MAS',
    'agem': 'AGEM',
    'l2': 'L2',
    'vcl': 'VCL',
    'fine_tuning': 'Fine-tuning',
    'perfect_memory': 'Perfect Memory',

    'pitfall': 'Pitfall',
    'arms_dealer': 'Arms Dealer',
    'hide_and_seek': 'Hide and Seek',
    'floor_is_lava': 'Floor is Lava',
    'chainsaw': 'Chainsaw',
    'raise_the_roof': 'Raise the Roof',
    'run_and_gun': 'Run and Gun',
    'health_gathering': 'Health Gathering',

    'obstacles': 'Obstacles',
    'green': 'Green',
    'resized': 'Resized',
    'invulnerable': 'Monsters',
    'default': 'Default',
    'red': 'Red',
    'blue': 'Blue',
    'shadows': 'Shadows',

    'success': 'Success',
    'kills': 'Kill Count',
    'ep_length': 'Frames Alive',
    'arms_dealt': 'Weapons Delivered',
    'distance': 'Distance',

    'Action 0': 'NO-OP',
    'Action 1': 'EXECUTE',
    'Action 2': 'MOVE_FORWARD',
    'Action 3': 'MOVE_FORWARD, EXECUTE',
    'Action 4': 'TURN_RIGHT',
    'Action 5': 'TURN_RIGHT, EXECUTE',
    'Action 6': 'TURN_RIGHT, MOVE_FORWARD',
    'Action 7': 'TURN_RIGHT, MOVE_FORWARD, EXECUTE',
    'Action 8': 'TURN_LEFT',
    'Action 9': 'TURN_LEFT, EXECUTE',
    'Action 10': 'TURN_LEFT, MOVE_FORWARD',
    'Action 11': 'TURN_LEFT, MOVE_FORWARD, EXECUTE',
}

SEQUENCES = {
    'CD4': ['default', 'red', 'blue', 'shadows'],
    'CD8': ['obstacles', 'green', 'resized', 'invulnerable', 'default', 'red', 'blue', 'shadows'],
    'CO4': ['chainsaw', 'raise_the_roof', 'run_and_gun', 'health_gathering'],
    'CO8': ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun',
            'health_gathering'],
    'COC': ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun',
            'health_gathering'],
}

METRICS = {
    'pitfall': 'distance',
    'arms_dealer': 'arms_dealt',
    'hide_and_seek': 'ep_length',
    'floor_is_lava': 'ep_length',
    'chainsaw': 'kills',
    'raise_the_roof': 'ep_length',
    'run_and_gun': 'kills',
    'health_gathering': 'ep_length',
    'default': 'kills',
}


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    n_actions = 12
    seeds = args.seeds
    sequence = args.sequence
    methods = args.methods
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    figsize = (6, 6) if n_envs == 4 else (12, 8)
    fig, ax = plt.subplots(len(methods), 1, figsize=figsize)
    max_steps = -np.inf
    iterations = args.task_length * n_envs
    test_env = args.test_env
    folder = f'test_{test_env}' if test_env is not None else 'train'
    cmap = plt.get_cmap('tab20c')

    for j, method in enumerate(methods):
        seed_data = np.empty((len(seeds), iterations, n_actions))
        seed_data[:] = np.nan
        for k, seed in enumerate(seeds):
            path = os.path.join(os.getcwd(), 'data', 'actions', sequence, method, folder, f'seed_{seed}.json')
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                data = json.load(f)
            steps = len(data)
            max_steps = max(max_steps, steps)
            seed_data[k, np.arange(steps)] = data

        # Find the mean actions over all the seeds
        mean = np.nanmean(seed_data, axis=0)
        mean = gaussian_filter1d(mean, sigma=5, axis=0)

        # Scale the values to add up to 1000 in each time step
        mean = mean / np.sum(mean, axis=1, keepdims=True) * 1000

        # Create a percent area stackplot with the values in mean
        ax[j].stackplot(np.arange(iterations), mean.T,
                        labels=[TRANSLATIONS[f'Action {i}'] for i in range(n_actions)],
                        colors=[cmap(i) for i in range(n_actions)])
        ax[j].tick_params(labelbottom=True)
        ax[j].set_title(TRANSLATIONS[method])
        ax[j].set_ylabel("Actions")

    add_coloured_task_labels(ax, envs, max_steps, n_envs)

    ax[-1].set_xlabel("Timesteps (K)")
    handles, labels = ax[-1].get_legend_handles_labels()
    n_cols = 4 if n_envs == 4 else 3
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=n_cols, fancybox=True, shadow=True,
               fontsize=11)

    title = TRANSLATIONS[SEQUENCES[sequence][test_env]] if test_env is not None else 'Train'
    fig.suptitle(f'        {sequence} - {title}', fontsize=16)
    bottom_adjust = 0.07 if n_envs == 4 else 0.13
    plt.tight_layout(rect=[0, bottom_adjust, 1, 1])

    file_path = 'plots/actions'
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(f'{file_path}/{sequence}_{title}.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default='CO8', choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        help="Name of the task sequence")
    parser.add_argument("--methods", type=str, default=['packnet', 'fine_tuning'],
                        choices=['packnet', 'mas', 'vcl', 'agem', 'l2', 'fine_tuning'])
    parser.add_argument("--test_env", type=int, help="Test environment ID of the actions to plot")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3, 4, 5], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
