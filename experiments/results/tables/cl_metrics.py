import argparse
import json
import numpy as np
import os
from scipy.stats import t
from typing import List

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
}

SEQUENCES = {
    'CD4': ['default', 'red', 'blue', 'shadows'],
    'CD8': ['obstacles', 'green', 'resized', 'invulnerable', 'default', 'red', 'blue', 'shadows'],
    'CO4': ['chainsaw', 'raise_the_roof', 'run_and_gun', 'health_gathering'],
    'CO8': ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun',
            'health_gathering'],
}

METHODS = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']


def calculate_data_at_the_end(data):
    return data[:, :, :, -10:].mean(axis=3)


def calculate_forgetting(data: np.ndarray):
    data_at_the_end = calculate_data_at_the_end(data)
    forgetting = (np.diagonal(data_at_the_end, axis1=1, axis2=2) - data_at_the_end[:, :, -1]).clip(0, np.inf)[:,
                 :-1].mean(axis=1)
    return data_at_the_end, forgetting


def calculate_performance(data: np.ndarray):
    data = data.mean(axis=3)
    data = np.triu(data)
    data[data == 0] = np.nan
    return np.nanmean(data, axis=(-1, -2))


def calc_metrics(metric: str, seeds: List[str], sequence: str, task_length: int, second_half: bool):
    envs = SEQUENCES[sequence]
    if second_half:
        envs = envs[len(envs) // 2:]
    n_envs = len(envs)
    iterations = n_envs * task_length
    methods = METHODS if n_envs == 4 or second_half else METHODS[:-1]  # Omit Perfect Memory for 8 env sequences
    cl_data = np.empty((len(methods), n_envs, n_envs, task_length))
    ci_data = np.empty((len(methods), n_envs, n_envs, task_length))
    cl_data[:] = np.nan
    ci_data[:] = np.nan
    dof = len(seeds) - 1
    significance = (1 - args.confidence) / 2
    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            seed_data = np.empty((len(seeds), n_envs, task_length))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), data, sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                if second_half:
                    data = data[len(data) // 2:]
                steps = len(data)
                data = np.pad(data, (0, iterations - steps), 'constant', constant_values=np.nan)
                data_per_task = np.array_split(data, n_envs)
                seed_data[k] = data_per_task

            mean = np.nanmean(seed_data, axis=0)
            std = np.nanstd(seed_data, axis=0)
            t_crit = np.abs(t.ppf(significance, dof))
            ci = std * t_crit / np.sqrt(len(seeds))
            cl_data[i][j] = mean
            ci_data[i][j] = ci
    performance = calculate_performance(cl_data)
    performance_ci = calculate_performance(ci_data)
    data_at_the_end, forgetting = calculate_forgetting(cl_data)
    _, forgetting_ci = calculate_forgetting(ci_data)
    # Print performance results
    print(f'\n\n{sequence}')
    print_results(performance, performance_ci, methods, "Performance")
    print_results(forgetting, forgetting_ci, methods, "Forgetting")

    return performance, performance_ci, forgetting, forgetting_ci


def print_results(metric_data: np.ndarray, ci: np.ndarray, methods: List[str], metric: str):
    print(f'\n{"Method":<20}{metric:<20}{"CI":<20}')
    for i, method in enumerate(methods):
        print(f'{method:<20}{metric_data[i]:<20.2f}{ci[i]:.2f}')


def normalize(metric_data, ci):
    joined_results = np.array((metric_data, ci))
    joined_results = joined_results / np.linalg.norm(joined_results)
    return joined_results


def calc_forward_transfer(performances, performance_ci, baseline):
    methods = METHODS[:-1]
    transfer = np.empty((len(methods), len(methods)))
    transfer_ci = np.empty((len(methods), len(methods)))

    # TODO calculate forward transfer

    print_results(transfer, transfer_ci, methods, "Transfer")
    return []


def get_baseline_data(seeds: List[str], task_length: int, sequence: str = 'CO8', set_metric: str = None) -> np.ndarray:
    scenarios = SEQUENCES[sequence]
    seed_data = np.empty((len(seeds), task_length * len(scenarios)))
    seed_data[:] = np.nan
    metric = 'success'
    for i, env in enumerate(scenarios):
        for k, seed in enumerate(seeds):
            path = f'{os.getcwd()}/single/sac/seed_{seed}/{env}_{metric}.json'
            with open(path, 'r') as f:
                data = json.load(f)[0: task_length]
            steps = len(data)
            start = i * task_length
            seed_data[k, np.arange(start, start + steps)] = data
    baseline_data = np.nanmean(seed_data, axis=0)
    return baseline_data


def main(cfg: argparse.Namespace) -> None:
    sequences = cfg.sequences

    performances = np.empty((len(sequences), len(METHODS)))
    performance_cis = np.empty((len(sequences), len(METHODS)))
    forgettings = np.empty((len(sequences), len(METHODS)))
    forgetting_cis = np.empty((len(sequences), len(METHODS)))
    for i, sequence in enumerate(sequences):
        performance, performance_ci, forgetting, forgetting_ci = calc_metrics(cfg.metric, cfg.seeds, sequence,
                                                                              cfg.task_length, cfg.second_half)
        performances[i] = np.pad(performance, (0, len(METHODS) - len(performance)), 'constant', constant_values=np.nan)
        performance_cis[i] = np.pad(performance_ci, (0, len(METHODS) - len(performance_ci)), 'constant',
                                    constant_values=np.nan)
        forgettings[i] = np.pad(forgetting, (0, len(METHODS) - len(forgetting)), 'constant', constant_values=np.nan)
        forgetting_cis[i] = np.pad(forgetting_ci, (0, len(METHODS) - len(forgetting_ci)), 'constant',
                                   constant_values=np.nan)
    performance = np.nanmean(performances, axis=0)
    performance_ci = np.nanmean(performance_cis, axis=0)
    forgetting = np.nanmean(forgettings, axis=0)
    forgetting_ci = np.nanmean(forgetting_cis, axis=0)

    # TODO calculate forward transfer for CO8 sequence
    baseline_data = get_baseline_data(cfg.seeds, cfg.task_length)
    transfer, transfer_ci = calc_forward_transfer(performances, performance_ci, baseline_data)

    print(f'\n\nAverage')
    print_results(performance, performance_ci, METHODS, "Performance")
    print_results(forgetting, forgetting_ci, METHODS, "Forgetting")
    print_results(transfer, transfer_ci, METHODS, "Performance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=str, nargs="+", required=True, choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        help="Sequences to evaluate")
    parser.add_argument("--seeds", type=str, nargs="+", default=['1', '2', '3'], help="Seeds to evaluate")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to calculate forgetting")
    parser.add_argument("--confidence", type=float, default=0.9, help="Confidence interval")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    parser.add_argument("--second_half", default=False, action='store_true', help="Only regard sequence 2nd half")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
