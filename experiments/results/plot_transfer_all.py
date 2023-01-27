import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import List

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

COLOR_SAC = '#C44E52'
COLORS = ['#1F77B4', '#55A868', '#4C72B0', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']
METHODS = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning']


def get_baseline_data(seeds: List[str], task_length: int, set_metric: str = None) -> np.ndarray:
    seed_data = np.empty((len(seeds), task_length * len(SCENARIOS)))
    seed_data[:] = np.nan
    for i, env in enumerate(SCENARIOS):
        metric = set_metric if set_metric else METRICS[env]
        for k, seed in enumerate(seeds):
            path = f'{os.getcwd()}/single/sac/seed_{seed}/{env}_{metric}.json'
            with open(path, 'r') as f:
                data = json.load(f)[0: task_length]
            steps = len(data)
            start = i * task_length
            seed_data[k, np.arange(start, start + steps)] = data
    baseline_data = np.nanmean(seed_data, axis=0)
    baseline_data = gaussian_filter1d(baseline_data, sigma=2)
    return baseline_data


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    seeds = ['1', '2', '3']
    n_envs = len(SCENARIOS)
    n_methods = len(METHODS)
    fig, ax = plt.subplots(n_methods, 1, sharex=True, figsize=(9, 12))
    task_length = cfg.task_length
    iterations = task_length * n_envs

    baseline = get_baseline_data(seeds, task_length, cfg.metric)

    for i, method in enumerate(METHODS):
        seed_data = np.empty((len(seeds), iterations))
        seed_data[:] = np.nan
        for j, env in enumerate(SCENARIOS):
            metric = cfg.metric if cfg.metric else METRICS[env]
            for k, seed in enumerate(seeds):
                path = f'{os.getcwd()}/{cfg.sequence}/{method}/seed_{seed}/{env}_{metric}.json'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = j * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    seed_data[k, np.arange(task_start, task_start + steps)] = data

        mean = np.nanmean(seed_data, axis=0)
        mean = gaussian_filter1d(mean, sigma=2)

        ax[i].plot(mean, label=TRANSLATIONS[method], color=COLORS[i])
        ax[i].plot(baseline, label=TRANSLATIONS['sac'], color=COLOR_SAC)
        ax[i].tick_params(labelbottom=True)
        ax[i].fill_between(np.arange(iterations), mean, baseline, where=(mean < baseline), alpha=0.2, color=COLOR_SAC, interpolate=True)
        ax[i].fill_between(np.arange(iterations), mean, baseline, where=(mean >= baseline), alpha=0.2, color=COLORS[i], interpolate=True)

        ax[i].set_ylabel(TRANSLATIONS[metric], fontsize=11)
        ax[i].set_title(TRANSLATIONS[method], fontsize=11)

    env_names = [TRANSLATIONS[e] for e in SCENARIOS]
    env_name_locations = np.arange(0 + task_length // 2, iterations + task_length // 2, task_length)

    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(env_name_locations)
    ax2.set_xticklabels(env_names, fontsize=9)
    ax2.tick_params(axis='both', which='both', length=0)


    handles, labels = [], []
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    ax[-1].set_xlabel("Timesteps (K)")
    ax[-1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=n_methods + 1, fancybox=True, shadow=True)
    fig.tight_layout()
    plt.savefig(f'plots/transfer.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, nargs="+", default='CO8', choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'])
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to plot")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
