import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t

from coom.utils.utils import str2bool

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
    'default': 'kills',
}

PLOT_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']
METHODS = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    seeds = ['1', '2', '3']
    sequence = args.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    metric = None
    figsize = (6, 6) if n_envs == 4 else (7, 13)
    share_y = sequence in ['CD4', 'CD8']
    fig, ax = plt.subplots(n_envs, 1, sharex=True, sharey=share_y, figsize=figsize)
    max_steps = -np.inf
    iterations = args.task_length * n_envs
    methods = METHODS if n_envs == 4 else METHODS[:-1]
    dof = len(seeds) - 1
    significance = (1 - args.confidence) / 2

    for i, env in enumerate(envs):
        for j, method in enumerate(methods):
            metric = args.metric if args.metric else METRICS[env] if env in METRICS else 'kills'
            seed_data = np.empty((len(seeds), iterations))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                steps = len(data)
                max_steps = max(max_steps, steps)
                seed_data[k, np.arange(steps)] = data

            mean = np.nanmean(seed_data, axis=0)
            mean = gaussian_filter1d(mean, sigma=2)
            std = np.nanstd(seed_data, axis=0)
            std = gaussian_filter1d(std, sigma=2)

            t_crit = np.abs(t.ppf(significance, dof))
            ci = std * t_crit / np.sqrt(len(seeds))

            ax[i].plot(mean, label=TRANSLATIONS[method], color=PLOT_COLORS[j])
            ax[i].tick_params(labelbottom=True)
            ax[i].fill_between(np.arange(iterations), mean - ci, mean + ci, alpha=0.2, color=PLOT_COLORS[j])

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[env])
        ax[i].yaxis.set_label_coords(-0.07, 0.5)

    env_steps = max_steps // n_envs
    task_indicators = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

    tick_labels = ['Task 1', 'Task 2', 'Task 3', 'Task 4'] if n_envs == 4 \
        else ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6', 'Task 7', 'Task 8']
    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(task_indicators)
    ax2.set_xticklabels(tick_labels)
    ax2.tick_params(axis='both', which='both', length=0)

    ax[-1].set_xlabel("Timesteps (K)")
    if args.plot_legend:
        handles, labels = ax[-1].get_legend_handles_labels()
        n_cols = 4 if n_envs == 4 else 6
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=n_cols, fancybox=True, shadow=True, fontsize=11)
        bottom_adjust = 0.07 if n_envs == 4 else 0.02
        plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
    else:
        plt.tight_layout()
    plot_name = f'{sequence}_envs_{metric}' if args.metric else sequence
    plt.savefig(f'plots/{plot_name}.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, choices=['CD4', 'CO4', 'CD8', 'CO8'],
                        help="Name of the task sequence")
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to plot")
    parser.add_argument("--confidence", type=float, default=0.9, help="Confidence interval")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    parser.add_argument("--plot_legend", type=str2bool, default=True, help="Whether to plot the legend")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
