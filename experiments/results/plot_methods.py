import json
import os
from matplotlib import pyplot as plt

from experiments.results.common import *
from experiments.results.common import plot_curve


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-paper')
    colors = COLORS[cfg.sequence]
    seeds = cfg.seeds
    sequence = cfg.sequence
    metric = cfg.metric
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    methods = METHODS if n_envs == 4 else METHODS[:-1]
    fig, ax = plt.subplots(len(methods), 1, sharey='all', sharex='all', figsize=(9, 12))
    max_steps = -np.inf
    iterations = cfg.task_length * n_envs

    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            seed_data = np.empty((len(seeds), iterations))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), 'data', sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                steps = len(data)
                max_steps = max(max_steps, steps)
                seed_data[k, np.arange(steps)] = data
            plot_curve(ax, cfg.confidence, colors[j], TRANSLATIONS[env], i, iterations, seed_data, len(seeds))

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[method], fontsize=12)

    add_task_labels(ax, envs, sequence, max_steps, n_envs)

    n_cols = n_envs if n_envs == 4 else n_envs // 2
    ax[-1].set_xlabel("Timesteps (K)", fontsize=11)
    ax[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), ncol=n_cols, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(f'plots/method/{sequence}_{metric}.png')
    plt.show()


if __name__ == "__main__":
    parser = common_args()
    main(parser.parse_args())
