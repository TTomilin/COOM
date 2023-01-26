import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import List
from scipy.stats import t

from experiments.results.cl_metrics import print_results

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
    # baseline_data = gaussian_filter1d(baseline_data, sigma=2)
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
    ft_all = [[] for i in range(len(METHODS))]
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
            data = np.nanmean(seed_data, axis=0)
            baseline_performance = baseline[np.arange(task_start, task_start + steps)]
            cl_performance = data[np.arange(task_start, task_start + steps)]
            
            baseline_mean = np.nanmean(baseline_performance)
            mean = np.nanmean(cl_performance)
            ft_i = (mean - baseline_mean) / (1-baseline_mean)
            ft_all[i].append(ft_i)
    mean = np.nanmean(np.array(ft_all), -1)
    std = np.nanstd(np.array(ft_all), -1)
    t_crit = np.abs(t.ppf((1-0.9)/2, len(seeds)-1))
    ci = std * t_crit / np.sqrt(len(seeds))
    print_results(mean, ci, METHODS, "forward transfor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, nargs="+", default='CO8', choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'])
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to plot")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
