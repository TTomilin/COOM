import os

from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, n_envs, methods = cfg.seeds, cfg.metric, cfg.n_envs, cfg.methods
    fig, ax = plt.subplots(len(methods), 1, sharey='all', sharex='all', figsize=(12, 7))
    max_steps = -np.inf
    iterations = cfg.task_length * n_envs

    for i, method in enumerate(methods):
        for s, sequence in enumerate(cfg.sequences):
            envs = SEQUENCES[sequence]
            envs = envs[:n_envs]
            colors = COLORS[sequence]
            seed_data = np.empty((len(seeds), iterations))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), '../data', sequence, method, f'seed_{seed}', 'train_success.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                data = data[:iterations]
                steps = len(data)
                max_steps = max(max_steps, steps)
                seed_data[k, np.arange(steps)] = data

            plot_curve(ax[i], cfg.confidence, colors[s], sequence, iterations, seed_data, len(seeds))

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[method])

    add_task_labels(ax[0], envs, max_steps, n_envs)
    plot_name = f'training/{"vs".join(cfg.sequences)}_{n_envs}envs'
    plot_and_save(ax=ax[-1], plot_name=plot_name, n_col=8, legend_anchor=-0.7)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--n_envs", type=int, default=8, help="Number of environments to plot")
    main(parser.parse_args())
