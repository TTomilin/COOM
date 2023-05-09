import os

from experiments.results.common import *
from experiments.results.common import plot_curve


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds, metric, sequence = cfg.seeds, cfg.metric, cfg.sequence
    colors = COLORS[sequence]
    envs = SEQUENCES[sequence]
    n_seeds, n_envs = len(seeds), len(envs)
    methods = METHODS if n_envs == 4 else METHODS[:-1]
    fig, ax = plt.subplots(len(methods), 1, sharey='all', sharex='all', figsize=(9, 12))
    max_steps = -np.inf
    iterations = cfg.task_length * n_envs

    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            seed_data = np.empty((n_seeds, iterations))
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
            plot_curve(ax[i], cfg.confidence, colors[j], TRANSLATIONS[env], iterations, seed_data, n_seeds)

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[method], fontsize=12)

    add_coloured_task_labels(ax[0], envs, sequence, max_steps, n_envs)
    n_cols = n_envs if n_envs == 4 else n_envs // 2
    plot_name = f'method/{sequence}_{metric}'
    plot_and_save(ax=ax[-1], plot_name=plot_name, n_col=n_cols, legend_anchor=-0.8)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
