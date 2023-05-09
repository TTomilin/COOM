import os

from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-paper')
    plt.rcParams['axes.grid'] = True
    colors = COLORS[cfg.sequence]
    methods, seeds, folders, sequence = cfg.methods, cfg.seeds, cfg.folders, cfg.sequence
    envs = SEQUENCES[sequence]
    n_envs, n_seeds, n_methods = len(envs), len(seeds), len(methods)
    fig_size = (12, 7) if n_methods < 4 else (12, 10)
    fig, ax = plt.subplots(n_methods, 1, sharey='all', sharex='all', figsize=fig_size)
    max_steps = -np.inf
    iterations = cfg.task_length * n_envs

    for i, method in enumerate(methods):
        for j, folder in enumerate(folders):
            seed_data = np.empty((n_envs, n_seeds, iterations))
            seed_data[:] = np.nan
            for e, env in enumerate(envs):
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

            plot_curve(ax, cfg.confidence, colors[j], TRANSLATIONS[folder], i, iterations, seed_data, n_seeds * n_envs,
                       agg_axes=(0, 1))

        ax[i].set_ylabel('Average Success')
        ax[i].set_title(TRANSLATIONS[method])

    add_task_labels(ax, envs, iterations, n_envs)
    plot_and_save(ax[-1], cfg.plot_name, len(folders), cfg.legend_anchor)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--folders", type=str, required=True, nargs='+', help="Names of the folders")
    parser.add_argument("--plot_name", type=str, required=True, help="Names of the plot")
    parser.add_argument("--legend_anchor", type=float, default=0, help="How much to lower the legend")
    parser.add_argument("--methods", type=str, nargs='+', default=['packnet', 'l2', 'mas', 'agem'],
                        help="Names of the methods")
    main(parser.parse_args())
