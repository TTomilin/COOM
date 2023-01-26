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
    'sac': 'SAC',

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

COLORS = ['#C44E52', '#55A868']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    seeds = ['1', '2', '3']
    sequences = args.sequences
    n_envs = len(SCENARIOS)
    metric = None
    n_rows = 2
    n_cols = int(np.ceil(n_envs / n_rows))
    # fig, ax = plt.subplots(n_rows, n_cols, sharex=True, figsize=(5, 8))
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True, figsize=(8, 4))
    task_length = args.task_length

    for i, env in enumerate(SCENARIOS):
        row = i % n_cols
        col = i // n_cols
        reference = None
        for j, sequence in enumerate(sequences):
            method = 'sac' if sequence == 'single' else args.method
            metric = args.metric if args.metric else METRICS[env]
            seed_data = np.empty((len(seeds), task_length))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = 0 if sequence == 'single' else i * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                steps = len(data)
                seed_data[k, np.arange(steps)] = data

            mean = np.nanmean(seed_data, axis=0)
            mean = gaussian_filter1d(mean, sigma=2)

            ax[col, row].plot(mean, label=TRANSLATIONS[method], color=COLORS[j])
            ax[col, row].tick_params(labelbottom=True)
            ax[col, row].ticklabel_format(style='sci', axis='y', scilimits=(0, 4))
            if reference is None:
                reference = mean
            else:
                ax[col, row].fill_between(np.arange(task_length), mean, reference, where=(mean < reference), alpha=0.2, color=COLORS[0], interpolate=True)
                ax[col, row].fill_between(np.arange(task_length), mean, reference, where=(mean >= reference), alpha=0.2, color=COLORS[1], interpolate=True)

        ax[col, row].set_ylabel(TRANSLATIONS[metric], fontsize=11)
        ax[col, row].set_title(TRANSLATIONS[env], fontsize=11)
        ax[col, row].yaxis.set_label_coords(-0.26, 0.5)

    main_ax = fig.add_subplot(1, 1, 1, frameon=False)
    main_ax.get_xaxis().set_ticks([])
    main_ax.get_yaxis().set_ticks([])
    main_ax.set_xlabel('Timesteps (K)')
    main_ax.xaxis.labelpad = 25

    handles, labels = ax[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fancybox=True, shadow=True)
    fig.tight_layout()
    plt.savefig(f'plots/transfer_{method}.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=str, nargs="+", default=['single', 'CO8'])
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to plot")
    parser.add_argument("--method", type=str, default='packnet', help="CL method name")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
