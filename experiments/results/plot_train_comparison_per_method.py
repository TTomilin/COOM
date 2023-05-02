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

    'obstacles': 'Obstacles',
    'green': 'Green',
    'resized': 'Resized',
    'invulnerable': 'Monsters',
    'default': 'Default',
    'red': 'Red',
    'blue': 'Blue',
    'shadows': 'Shadows',

    'success': 'Success',
    'kills': 'Score',
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

LINE_STYLES = ['-', '--', ':', '-.']

METHODS = ['packnet', 'l2', 'mas']


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    seeds = ['1', '2', '3']
    metric = cfg.metric
    n_envs = args.n_envs
    fig, ax = plt.subplots(len(METHODS), 1, sharey=True, sharex=True, figsize=(12, 7))
    env_names = [TRANSLATIONS[e] for e in SEQUENCES[args.sequences[0]]]
    env_names = env_names[:n_envs]
    max_steps = -np.inf
    iterations = cfg.task_length * n_envs
    dof = len(seeds) - 1
    significance = (1 - cfg.confidence) / 2

    for i, method in enumerate(METHODS):
        for s, sequence in enumerate(args.sequences):
            envs = SEQUENCES[sequence]
            envs = envs[:n_envs]
            colors = COLORS[sequence]
            seed_data = np.empty((len(seeds), iterations))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), sequence, method, f'seed_{seed}', 'train_success.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                data = data[:iterations]
                steps = len(data)
                max_steps = max(max_steps, steps)
                seed_data[k, np.arange(steps)] = data

            mean = np.nanmean(seed_data, axis=0)
            mean = gaussian_filter1d(mean, sigma=2)
            std = np.nanstd(seed_data, axis=0)
            std = gaussian_filter1d(std, sigma=2)

            t_crit = np.abs(t.ppf(significance, dof))
            ci = std * t_crit / np.sqrt(len(seeds))

            ax[i].plot(mean, label=sequence, color=colors[s])
            ax[i].tick_params(labelbottom=True)
            ax[i].fill_between(np.arange(iterations), mean - ci, mean + ci, alpha=0.2, color=colors[s])

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[method])

    env_steps = max_steps // n_envs
    env_name_locations = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(env_name_locations)
    ax2.set_xticklabels(env_names)
    ax2.tick_params(axis='both', which='both', length=0)

    ax[-1].set_xlabel("Timesteps (K)")
    ax[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=8, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'plots/training_{metric}_{"vs".join(args.sequences)}_{n_envs}envs.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=str, default=['CO8', 'COC'], nargs="+",
                        choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        help="Names of the task sequences")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to plot")
    parser.add_argument("--confidence", type=float, default=0.9, help="Confidence interval")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    parser.add_argument("--n_envs", type=int, default=8, help="Number of environments to plot")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
