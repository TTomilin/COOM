import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t

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

    'success': 'Success',
    'kills': 'Kill Count',
    'ep_length': 'Frames Alive',
    'arms_dealt': 'Weapons Delivered',
    'distance': 'Distance',

    'single': 'Single',
    'COC': 'COC',
    'CO8': 'CO8',
}

SCENARIOS = ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun',
             'health_gathering']

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
LINE_STYLES = ['-', '--', ':', '-.']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-paper')
    seeds = ['1', '2', '3']
    sequences = args.sequences
    n_envs = len(SCENARIOS)
    metric = None
    n_rows = 2
    n_cols = int(np.ceil(n_envs / n_rows))
    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', figsize=(10, 4))
    max_steps = -np.inf
    task_length = args.task_length
    dof = len(seeds) - 1
    significance = (1 - args.confidence) / 2

    for i, env in enumerate(SCENARIOS):
        row = i % n_cols
        col = i // n_cols
        for j, sequence in enumerate(sequences):
            for l, method in enumerate(args.methods):
                metric = args.metric if args.metric else METRICS[env]
                seed_data = np.empty((len(seeds), task_length))
                seed_data[:] = np.nan
                for k, seed in enumerate(seeds):
                    path = os.path.join(os.getcwd(), 'data', sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                    if not os.path.exists(path):
                        continue
                    with open(path, 'r') as f:
                        task_start = i * task_length
                        data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    max_steps = max(max_steps, steps)
                    seed_data[k, np.arange(steps)] = data

                mean = np.nanmean(seed_data, axis=0)
                mean = gaussian_filter1d(mean, sigma=2)
                std = np.nanstd(seed_data, axis=0)
                std = gaussian_filter1d(std, sigma=2)

                t_crit = np.abs(t.ppf(significance, dof))
                ci = std * t_crit / np.sqrt(len(seeds))

                ax[col, row].plot(mean, label=f'{TRANSLATIONS[method]} ({TRANSLATIONS[sequence]})', color=PLOT_COLORS[l], linestyle=LINE_STYLES[j])
                ax[col, row].tick_params(labelbottom=True)
                ax[col, row].fill_between(np.arange(task_length), mean - ci, mean + ci, alpha=0.3, color=PLOT_COLORS[l])

        ax[col, row].set_ylabel(TRANSLATIONS[metric])
        ax[col, row].set_title(TRANSLATIONS[env])
        ax[col, row].yaxis.set_label_coords(-0.2, 0.5)
        ax[col, row].ticklabel_format(axis='y', style='sci', scilimits=(0, 5))

    main_ax = fig.add_subplot(1, 1, 1, frameon=False)
    main_ax.get_xaxis().set_ticks([])
    main_ax.get_yaxis().set_ticks([])
    main_ax.set_xlabel('Timesteps (K)')
    main_ax.xaxis.labelpad = 25

    handles, labels = ax[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=6, fancybox=True, shadow=True)
    fig.tight_layout()
    plt.savefig('plots/COC/train_per_env.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=str, nargs="+", default=['CO8', 'COC'])
    parser.add_argument("--methods", type=str, nargs="+", choices=['packnet', 'vcl', 'mas', 'agem', 'l2', 'fine_tuning'])
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to plot")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence interval")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
