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
    'CO4': ['#55A868', '#C44E52', '#4C72B0', '#8172B2'],
    'CD8': ['#64B5CD', '#55A868', '#777777', '#8172B2', '#CCB974', '#C44E52', '#4C72B0', '#917113'],
    'CO8': ['#64B5CD', '#55A868', '#777777', '#8172B2', '#CCB974', '#C44E52', '#4C72B0', '#917113'],
    'COC': ['#64B5CD', '#55A868', '#777777', '#8172B2', '#CCB974', '#C44E52', '#4C72B0', '#917113'],
}

METHODS = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    seeds = ['1', '2', '3']
    sequences = cfg.sequences
    n_sequences = len(sequences)
    metric = cfg.metric
    fig, ax = plt.subplots(n_sequences, 1, sharey='all', sharex='all', figsize=(10, 6))
    max_steps = -np.inf
    n_seeds = len(seeds)

    for l, sequence in enumerate(sequences):
        envs = SEQUENCES[sequence]
        n_envs = len(envs)
        methods = METHODS if n_envs == 4 else METHODS[:-1]
        iterations = cfg.task_length * n_envs
        dof = n_seeds * n_envs - 1
        significance = (1 - cfg.confidence) / 2
        env_names = [TRANSLATIONS[env] for env in envs]
        for i, method in enumerate(methods):
            seed_data = np.empty((n_envs, n_seeds, iterations))
            seed_data[:] = np.nan
            for j, env in enumerate(envs):
                for k, seed in enumerate(seeds):
                    path = os.path.join(os.getcwd(), 'data', sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                    if not os.path.exists(path):
                        continue
                    with open(path, 'r') as f:
                        data = json.load(f)
                    steps = len(data)
                    max_steps = max(max_steps, steps)
                    seed_data[j, k, np.arange(steps)] = data

            mean = np.nanmean(seed_data, axis=(0, 1))
            mean = gaussian_filter1d(mean, sigma=2)
            std = np.nanstd(seed_data, axis=(0, 1))
            std = gaussian_filter1d(std, sigma=2)

            t_crit = np.abs(t.ppf(significance, dof))
            ci = std * t_crit / np.sqrt(n_seeds * n_envs)

            ax[l].plot(mean, label=TRANSLATIONS[method])
            ax[l].tick_params(labelbottom=True)
            ax[l].fill_between(np.arange(iterations), mean - ci, mean + ci, alpha=0.2)

        ax[l].set_ylabel('Average Success')
        ax[l].set_title(sequence, fontsize=16)

        env_steps = max_steps // n_envs
        env_name_locations = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

        fontsize = 10 if n_envs == 4 else 9
        ax2 = ax[l].twiny()
        ax2.set_xlim(ax[l].get_xlim())
        ax2.set_xticks(env_name_locations)
        ax2.set_xticklabels(env_names, fontsize=fontsize)
        ax2.tick_params(axis='both', which='both', length=0)

        for xtick, color in zip(ax2.get_xticklabels(), COLORS[sequence]):
            xtick.set_color(color)
            xtick.set_fontweight('bold')

    ax[-1].set_xlabel("Timesteps (K)")
    ax[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=len(methods[0]), fancybox=True, shadow=True)
    plt.tight_layout()
    folder = 'plots/success'
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{"_".join(sequences)}.png')
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", type=str, nargs='+', required=True, choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        help="Name of the task sequences")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to plot")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence interval")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
