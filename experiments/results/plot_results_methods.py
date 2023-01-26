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
}

COLORS = {
    'CD4': ['#55A868', '#C44E52', '#4C72B0', '#8172B2'],
    'CO4': ['#4C72B0', '#55A868', '#C44E52', '#8172B2'],
    'CD8': ['#64B5CD', '#55A868', '#777777', '#8172B2', '#CCB974', '#C44E52', '#4C72B0', '#917113'],
    'CO8': ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113'],
    'COC': ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']
}

METHODS = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    colors = COLORS[cfg.sequence]
    seeds = ['1', '2', '3']
    sequence = cfg.sequence
    metric = cfg.metric
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    methods = METHODS if n_envs == 4 else METHODS[:-1]
    # figsize = (6, 12) if n_envs == 4 else (9, 13)
    fig, ax = plt.subplots(len(methods), 1, sharey=True, sharex=True, figsize=(9, 14))
    env_names = [TRANSLATIONS[e] for e in envs]
    max_steps = -np.inf
    iterations = cfg.task_length * n_envs
    dof = len(seeds) - 1
    significance = (1 - cfg.confidence) / 2

    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
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

            ax[i].plot(mean, label=TRANSLATIONS[env], color=colors[j])
            ax[i].tick_params(labelbottom=True)
            ax[i].fill_between(np.arange(iterations), mean - ci, mean + ci, alpha=0.2, color=colors[j])

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[method])

    n_envs = len(envs)
    env_steps = max_steps // n_envs
    env_name_locations = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

    fontsize = 10 if n_envs == 4 else 9
    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(env_name_locations)
    ax2.set_xticklabels(env_names, fontsize=fontsize)
    ax2.tick_params(axis='both', which='both', length=0)

    colors = [ax[0].get_lines()[i].get_color() for i in range(n_envs)]
    for xtick, color in zip(ax2.get_xticklabels(), colors):
        xtick.set_color(color)
        xtick.set_fontweight('bold')

    n_cols = n_envs if n_envs == 4 else n_envs // 2
    ax[-1].set_xlabel("Timesteps (K)")
    ax[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=n_cols, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'plots/{sequence}_methods_{metric}.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, choices=['CD4', 'CO4', 'CD8', 'CO8'],
                        help="Name of the task sequence")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to plot")
    parser.add_argument("--confidence", type=float, default=0.9, help="Confidence interval")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)