import json
import os
from matplotlib import pyplot as plt

from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-paper')
    seeds = args.seeds
    sequence = args.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    metric = None
    figsize = (8, 8) if n_envs == 4 else (10, 13)
    y_label_shift = -0.06 if n_envs == 4 else -0.04
    share_y = sequence in ['CD4', 'CD8'] or args.metric == 'success'
    fig, ax = plt.subplots(n_envs, 1, sharex='all', sharey=share_y, figsize=figsize)
    max_steps = -np.inf
    iterations = args.task_length * n_envs
    methods = METHODS if n_envs == 4 else METHODS[:-1]

    for i, env in enumerate(envs):
        for j, method in enumerate(methods):
            metric = args.metric if args.metric else METRICS[env] if env in METRICS else 'kills'
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

            plot_curve(ax, args.confidence, PLOT_COLORS[j], TRANSLATIONS[method], i, iterations, seed_data, len(seeds))

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[env])
        ax[i].yaxis.set_label_coords(y_label_shift, 0.5)

    add_task_labels(ax, envs, sequence, max_steps, n_envs)

    fontsize = 9 if n_envs == 4 else 12
    legend_anchor = -0.7 if n_envs == 4 else -1

    ax[-1].set_xlabel("Timesteps (K)", fontsize=11)
    ax[-1].legend(loc='lower center', bbox_to_anchor=(0.5, legend_anchor), ncol=len(methods), fancybox=True, shadow=True, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f'plots/env/{sequence}_{metric}.png')
    plt.show()


if __name__ == "__main__":
    parser = common_args()
    main(parser.parse_args())
