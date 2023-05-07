import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import t
from typing import List

from experiments.results.tables.cl_metrics import print_results

SCENARIOS = ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun',
             'health_gathering']

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
    return baseline_data


def get_cl_data(seeds, task_length, metric: str = 'success', sequence: str = 'CO8'):
    cl_data = np.empty((len(seeds), len(METHODS), task_length * len(SCENARIOS)))
    cl_data[:] = np.nan
    for k, seed in enumerate(seeds):
        for i, method in enumerate(METHODS):
            for j, env in enumerate(SCENARIOS):
                path = f'{os.getcwd()}/data/{sequence}/{method}/seed_{seed}/{env}_{metric}.json'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = j * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    cl_data[k, i, np.arange(task_start, task_start + steps)] = data
    return cl_data


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    seeds = ['1', '2', '3']
    n_seeds = len(seeds)
    task_length = cfg.task_length

    cl_data = get_cl_data(seeds, task_length, cfg.metric, cfg.sequence)
    baseline_data = get_baseline_data(seeds, task_length, cfg.metric)

    auc_cl = np.nanmean(cl_data, axis=-1)
    auc_baseline = np.nanmean(baseline_data, axis=-1)

    ft = (auc_cl - auc_baseline) / (1 - auc_baseline)
    ft_mean = np.nanmean(ft, 0)
    ft_std = np.nanstd(ft, 0)
    t_crit = np.abs(t.ppf((1 - 0.9) / 2, n_seeds - 1))
    ci = ft_std * t_crit / np.sqrt(n_seeds)
    print_results(ft_mean, ci, METHODS, "Forward Transfer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default='CO8', choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'])
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to plot")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
