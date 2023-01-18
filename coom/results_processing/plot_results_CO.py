import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

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
    'invulnerable': 'Invulnerable',
    'default': 'Default',
    'red': 'Red',
    'blue': 'Blue',
    'shadows': 'Shadows',

    'success': 'Success Rate',
    'kills': 'Kill Count',
    'ep_length': 'Frames Alive',
    'arms_dealt': 'Weapons Delivered',
    'distance': 'Distance',
}

SEQUENCES = {
    'CD4': ['default', 'red', 'blue', 'shadows'],
    'CD8': ['obstacles', 'green', 'resized', 'invulnerable', 'default', 'red', 'blue', 'shadows'],
    'CO4': ['chainsaw', 'raise_the_roof', 'run_and_gun', 'health_gathering'],
    'CO8': ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun',
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
}

PLOT_COLORS = ['#55A868', '#C44E52', '#4C72B0', '#8172B2']
METHODS = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    seeds = ['1', '2', '3']
    cl_data = {}

    sequence = args.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    metric = None
    figsize = (6, 7) if sequence == 'CO4' else (7, 13)
    fig, ax = plt.subplots(n_envs, 1, sharex=True, figsize=figsize)
    max_steps = -np.inf
    iterations = 800 if sequence in ['CD4', 'CO4'] else 1600

    for i, env in enumerate(envs):
        for j, method in enumerate(METHODS):
            metric = METRICS[env] if args.metric is None else args.metric
            seed_data = np.empty((len(seeds), iterations))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                # max_steps = max(max_steps, len(data))
                cl_data[f'{method}_{env}'] = data
                steps = len(data)
                max_steps = max(max_steps, steps)
                seed_data[k, np.arange(steps)] = data
                # print(f'{method}_{env}_{seed}: {len(steps)}')

            y = np.nanmean(seed_data, axis=0)
            y = gaussian_filter1d(y, sigma=2)
            ci = np.nanstd(seed_data, axis=0)
            ci = gaussian_filter1d(ci, sigma=2)
            ax[i].plot(y, label=TRANSLATIONS[method])
            ax[i].tick_params(labelbottom=True)
            ax[i].fill_between(np.arange(iterations), y - ci, y + ci, alpha=0.2)
            # print(f'{method}_{env} nan count: {np.isnan(y).sum()}')

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[env])

    env_steps = max_steps // n_envs
    task_indicators = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

    tick_labels = ['Task 1', 'Task 2', 'Task 3', 'Task 4'] if sequence == 'CO4' \
        else ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7', 'Task 8']
    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(task_indicators)
    ax2.set_xticklabels(tick_labels)
    ax2.tick_params(axis='both', which='both', length=0)

    ax[-1].set_xlabel("Timesteps (K)")
    handles, labels = ax[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=4, fancybox=True, shadow=True)
    bottom_adjust = 0.06 if sequence == 'CO4' else 0.04
    plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
    plt.savefig(f'plots/{sequence}.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, choices=['CD4', 'CO4', 'CD8', 'CO8'],
                        help="Name of the task sequence")
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to plot")
    parser.add_argument("--output_path", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
