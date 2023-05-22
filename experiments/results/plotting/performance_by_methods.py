from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds, metric, sequence = cfg.seeds, cfg.metric, cfg.sequence
    colors = COLORS[sequence]
    envs = SEQUENCES[sequence]
    n_seeds, n_envs = len(seeds), len(envs)
    methods = METHODS if n_envs == 4 else METHODS[:-1]
    fig, ax = plt.subplots(len(methods), 1, sharey='all', sharex='all', figsize=(11, 12))
    iterations = cfg.task_length * n_envs

    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            seed_data = get_data(env, iterations, method, metric, seeds, sequence)
            plot_curve(ax[i], cfg.confidence, colors[j], TRANSLATIONS[env], iterations, seed_data, n_seeds)

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[method], fontsize=12)

    add_coloured_task_labels(ax[0], envs, sequence, iterations, n_envs)
    n_cols = n_envs if n_envs == 4 else n_envs // 2
    plot_name = f'method/{sequence}_{metric}'
    plot_and_save(ax=ax[-1], plot_name=plot_name, n_col=n_cols, legend_anchor=-0.8)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
