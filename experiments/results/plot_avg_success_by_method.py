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

    'reg_critic': 'Critic Regularization',
    'no_reg_critic': 'No Critic Regularization',

    'single_head': 'Single Head',
    'multi_head': 'Multi Head',
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


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-paper')
    plt.rcParams['axes.grid'] = True
    seeds = args.seeds
    folders = args.folders
    sequence = cfg.sequence
    fig_size = (12, 7) if len(cfg.methods) < 4 else (12, 10)
    fig, ax = plt.subplots(len(cfg.methods), 1, sharey='all', sharex='all', figsize=fig_size)
    env_names = [TRANSLATIONS[e] for e in SEQUENCES[sequence]]
    n_envs = len(env_names)
    max_steps = -np.inf
    iterations = cfg.task_length * n_envs
    n_seeds = len(seeds)
    dof = n_seeds - 1
    significance = (1 - cfg.confidence) / 2

    for i, method in enumerate(cfg.methods):
        for folder in folders:
            seed_data = np.empty((n_envs, n_seeds, iterations))
            seed_data[:] = np.nan
            for e, env in enumerate(SEQUENCES[sequence]):
                for k, seed in enumerate(seeds):
                    path = f'{os.getcwd()}/data/{folder}/{sequence}/{method}/seed_{seed}/{env}_success.json'
                    if not os.path.exists(path):
                        print(f'Path {path} does not exist')
                        continue
                    with open(path, 'r') as f:
                        data = json.load(f)
                    data = data
                    steps = len(data)
                    max_steps = max(max_steps, steps)
                    seed_data[e, k, np.arange(steps)] = data

            mean = np.nanmean(seed_data, axis=(0, 1))
            mean = gaussian_filter1d(mean, sigma=2)
            std = np.nanstd(seed_data, axis=(0, 1))
            std = gaussian_filter1d(std, sigma=2)

            lb = np.quantile(mean, significance, interpolation="midpoint")
            ub = np.quantile(mean, 1 - significance, interpolation="midpoint")

            t_crit = np.abs(t.ppf(significance, dof))
            ci = std * t_crit / np.sqrt(n_seeds * n_envs)

            ax[i].plot(mean, label=TRANSLATIONS[folder])
            ax[i].tick_params(labelbottom=True)
            # ax[i].fill_between(np.arange(iterations), mean - ci, mean + ci, alpha=0.25)
            ax[i].fill_between(np.arange(iterations), lb, ub, alpha=0.25)

        ax[i].set_ylabel('Average Success')
        ax[i].set_title(TRANSLATIONS[method])

    env_steps = max_steps // n_envs
    env_name_locations = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(env_name_locations)
    ax2.set_xticklabels(env_names)
    ax2.tick_params(axis='both', which='both', length=0)

    legend_anchor = -0.7 if len(cfg.methods) < 4 else -0.8

    ax[-1].set_xlabel("Timesteps (K)", fontsize=13)
    ax[-1].legend(loc='lower center', bbox_to_anchor=(0.5, legend_anchor), ncol=len(folders), fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'plots/{args.plot_name}.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default='CO8', choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        help="Names of the task sequences")
    parser.add_argument("--methods", type=str, nargs='+', default=['packnet', 'l2', 'agem'], help="Names of the methods")
    parser.add_argument("--folders", type=str, required=True, nargs='+', help="Names of the folders")
    parser.add_argument("--plot_name", type=str, required=True, help="Names of the plot")
    parser.add_argument("--seeds", type=int, default=[1, 2, 3], help="")
    parser.add_argument("--confidence", type=float, default=0.8, help="Confidence interval")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
