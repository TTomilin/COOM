import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from typing import List

TRANSLATIONS = {
    'sac': 'SAC (reference)',
    'packnet': 'PackNet',
    'mas': 'MAS',
    'agem': 'AGEM',
    'ewc': 'EWC',
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

    'reg_critic': 'Critic Regularization',
    'no_reg_critic': 'No Critic Regularization',

    'single_head': 'Single Head',
    'multi_head': 'Multi Head',

    'single': 'Single',
    'CD4': 'CD4',
    'CD8': 'CD8',
    'CO4': 'CO4',
    'CO8': 'CO8',
    'COC': 'COC',

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

COLORS = {
    'CD4': ['#55A868', '#C44E52', '#4C72B0', '#8172B2'],
    'CO4': ['#4C72B0', '#55A868', '#C44E52', '#8172B2'],
    'CD8': ['#64B5CD', '#55A868', '#777777', '#8172B2', '#CCB974', '#C44E52', '#4C72B0', '#917113'],
    'CO8': ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113'],
    'COC': ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']
}

PLOT_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']

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

ENVS = {
    'CO4': 'default',
    'CO8': 'default',
    'COC': 'hard',
}

SEPARATE_STORAGE_TAGS = ['REG_CRITIC', 'NO_REG_CRITIC', 'SINGLE_HEAD']
FORBIDDEN_TAGS = ['SINGLE_HEAD', 'REG_CRITIC', 'NO_REG_CRITIC', 'SPARSE', 'TEST']
LINE_STYLES = ['-', '--', ':', '-.']
METHODS = ['packnet', 'mas', 'agem', 'l2', 'ewc', 'vcl', 'fine_tuning', 'perfect_memory']
KERNEL_SIGMA = 3
INTERVAL_INTENSITY = 0.25
CRITICAL_VALUES = {
    0.9: 1.833,
    0.95: 1.96,
    0.99: 2.576
}


def common_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default='CO8', choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        help="Name of the task sequence")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1, 2, 3, 4, 5], help="Seed(s) of the run(s) to plot")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to store/plot")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    parser.add_argument("--test_envs", type=int, nargs='+', help="Test environment ID of the actions to download/plot")
    return parser


def common_plot_args() -> argparse.ArgumentParser:
    parser = common_args()
    parser.add_argument("--method", type=str, default='packnet', help="CL method name")
    parser.add_argument("--confidence", type=float, default=0.95, choices=[0.9, 0.95, 0.99], help="Confidence interval")
    parser.add_argument("--sequences", type=str, default=['CO8', 'COC'], choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        nargs='+', help="Name of the task sequences")
    parser.add_argument("--methods", type=str, nargs="+",
                        choices=['packnet', 'vcl', 'mas', 'ewc', 'agem', 'l2', 'fine_tuning'])
    return parser


def common_dl_args() -> argparse.ArgumentParser:
    parser = common_args()
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--method", type=str, help="Optional filter by CL method")
    parser.add_argument("--type", type=str, default='test', choices=['train', 'test'], help="Type of data to download")
    parser.add_argument("--wandb_tags", type=str, nargs='+', default=[], help="WandB tags to filter runs")
    parser.add_argument("--overwrite", default=False, action='store_true', help="Overwrite existing files")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="List of runs that shouldn't be filtered out")
    return parser


def add_task_labels(ax, envs: List[str], iterations: int, n_envs: int):
    env_steps = iterations // n_envs
    task_indicators = np.arange(0 + env_steps // 2, iterations + env_steps // 2, env_steps)
    fontsize = 10 if n_envs == 4 else 9
    tick_labels = [TRANSLATIONS[env] for env in envs]
    ax_twin = ax.twiny()
    ax_twin.set_xlim(ax.get_xlim())
    ax_twin.set_xticks(task_indicators)
    ax_twin.set_xticklabels(tick_labels, fontsize=fontsize)
    ax_twin.tick_params(axis='both', which='both', length=0)
    return ax_twin


def add_coloured_task_labels(ax: np.ndarray, envs: List[str], sequence: str, max_steps: int, n_envs: int):
    ax_twin = add_task_labels(ax, envs, max_steps, n_envs)
    for xtick, color in zip(ax_twin.get_xticklabels(), COLORS[sequence]):
        xtick.set_color(color)
        xtick.set_fontweight('bold')


def plot_curve(ax, confidence: float, color, label: str, iterations: int, seed_data: np.ndarray, n_seeds: int,
               agg_axes=0, linestyle='-'):
    mean = np.nanmean(seed_data, axis=agg_axes)
    std = np.nanstd(seed_data, axis=agg_axes)
    mean = gaussian_filter1d(mean, sigma=KERNEL_SIGMA)
    std = gaussian_filter1d(std, sigma=KERNEL_SIGMA)
    ci = CRITICAL_VALUES[confidence] * std / np.sqrt(n_seeds)
    ax.plot(mean, label=label, color=color, linestyle=linestyle)
    ax.tick_params(labelbottom=True)
    ax.fill_between(np.arange(iterations), mean - ci, mean + ci, alpha=INTERVAL_INTENSITY, color=color)


def add_main_ax(fig):
    main_ax = fig.add_subplot(1, 1, 1, frameon=False)
    main_ax.get_xaxis().set_ticks([])
    main_ax.get_yaxis().set_ticks([])
    main_ax.set_xlabel('Timesteps (K)', fontsize=11)
    main_ax.xaxis.labelpad = 25
    return main_ax


def get_cl_method(run):
    method = run.config["cl_method"]
    if not method:
        method = 'perfect_memory' if run.config['buffer_type'] == 'reservoir' else 'fine_tuning'
    return method


def suitable_run(run, args: argparse.Namespace) -> bool:
    # Check whether the provided CL sequence corresponds to the run
    if args.sequence not in run.url:
        return False
    # Check whether the provided method corresponds to the run
    if args.method and args.method != get_cl_method(run):
        return False
    # Load the configuration of the run
    config = json.loads(run.json_config)
    # Check whether the wandb tags are suitable
    if 'wandb_tags' in config:
        tags = config['wandb_tags']['value']
        # Check whether the run includes one of the provided tags
        if args.wandb_tags and not any(tag in tags for tag in args.wandb_tags):
            return False
        # Check whether the run includes one of the forbidden tags which is not in the provided tags
        if any(tag in tags for tag in FORBIDDEN_TAGS) and not any(tag in tags for tag in args.wandb_tags):
            return False
    # Check whether the run corresponds to one of the provided seeds
    if args.seeds:
        if 'seed' not in config:
            return False
        seed = config['seed']['value']
        if seed not in args.seeds:
            return False
    if args.method:
        method = get_cl_method(run)
        if method != args.method:
            return False
    # Include only finished runs and the specified crashed/failed ones
    if any(logs in run.name for logs in args.include_runs) or run.state == "finished":
        # All filters have been passed
        return True
    return False


def plot_and_save(ax, plot_name: str, n_col: int, legend_anchor: float = 0.0, fontsize: int = 11) -> None:
    ax.set_xlabel("Timesteps (K)", fontsize=fontsize)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, legend_anchor), ncol=n_col, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'plots/{plot_name}.png')
    plt.show()


def get_baseline_data(sequence: str, scenarios: List[str], seeds: List[str], task_length: int,
                      set_metric: str = None) -> np.ndarray:
    seed_data = np.empty((len(seeds), task_length * len(scenarios)))
    seed_data[:] = np.nan
    baseline_type = 'single_hard' if sequence == 'COC' else 'single'
    for i, env in enumerate(scenarios):
        metric = set_metric if set_metric else METRICS[env]
        for k, seed in enumerate(seeds):
            path = f'{os.getcwd()}/data/{baseline_type}/sac/seed_{seed}/{env}_{metric}.json'
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                data = json.load(f)[0: task_length]
            steps = len(data)
            start = i * task_length
            seed_data[k, np.arange(start, start + steps)] = data
    baseline_data = np.nanmean(seed_data, axis=0)
    return baseline_data


def get_action_data(folder: str, iterations: int, method: str, n_actions: int, seeds: List[int], sequence: str,
                    scale=True, ep_time_steps=1000, sigma=5):
    data = np.empty((len(seeds), iterations, n_actions))
    data[:] = np.nan
    for k, seed in enumerate(seeds):
        path = os.path.join(os.getcwd(), 'data', 'actions', sequence, method, folder, f'seed_{seed}.json')
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            seed_data = json.load(f)
            data[k, np.arange(len(seed_data))] = seed_data
    mean = np.nanmean(data, axis=0)
    mean = gaussian_filter1d(mean, sigma=sigma, axis=0)
    if scale:
        mean = mean / np.sum(mean, axis=1, keepdims=True) * ep_time_steps
    return mean


def get_data(env: str, iterations: int, method: str, metric: str, seeds: List[int], sequence: str):
    data = np.empty((len(seeds), iterations))
    data[:] = np.nan
    for k, seed in enumerate(seeds):
        path = os.path.join(os.getcwd(), 'data', sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
        if not os.path.exists(path):
            print(f'Path {path} does not exist')
            continue
        with open(path, 'r') as f:
            seed_data = json.load(f)
            data[k, np.arange(len(seed_data))] = seed_data
    return data


def get_data_per_env(envs: List[str], iterations: int, method: str, metric: str, seeds: List[int], sequence: str,
                     folder: str) -> np.ndarray:
    seed_data = np.empty((len(envs), len(seeds), iterations))
    seed_data[:] = np.nan
    for e, env in enumerate(envs):
        for k, seed in enumerate(seeds):
            path = os.path.join(os.getcwd(), 'data', folder, sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
            if not os.path.exists(path):
                print(f'Path {path} does not exist')
                continue
            with open(path, 'r') as f:
                data = json.load(f)
                seed_data[e, k, np.arange(len(data))] = data
    return seed_data
