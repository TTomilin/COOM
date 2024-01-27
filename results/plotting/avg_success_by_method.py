from results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    methods, seeds, folders, sequence, metric = cfg.methods, cfg.seeds, cfg.folders, cfg.sequence, cfg.metric
    envs = SEQUENCES[sequence]
    n_envs, n_seeds, n_methods = len(envs), len(seeds), len(methods)
    fig_size = (12, 10) if n_methods < 5 else (12, 13) if n_methods == 5 else (12, 14)
    fig, ax = plt.subplots(n_methods, 1, sharey='all', sharex='all', figsize=fig_size)
    iterations = cfg.task_length * n_envs

    for i, method in enumerate(methods):
        for j, folder in enumerate(folders):
            data = get_data_per_env(envs, iterations, method, metric, seeds, sequence, folder)
            plot_curve(ax[i], cfg.confidence, PLOT_COLORS[j], TRANSLATIONS[folder], iterations, data, n_seeds * n_envs,
                       agg_axes=(0, 1))

        ax[i].set_ylabel('Average Success')
        ax[i].set_title(TRANSLATIONS[method])
        ax[i].set_xlim(0, iterations)
        ax[i].set_ylim(0, 1)

    add_task_labels(ax[0], envs, iterations, n_envs)
    plot_and_save(ax[-1], cfg.plot_name, len(folders), cfg.legend_anchor, h_pad=cfg.h_pad)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--folders", type=str, required=True, nargs='+', help="Names of the folders")
    parser.add_argument("--plot_name", type=str, required=True, help="Names of the plot")
    parser.add_argument("--legend_anchor", type=float, default=0, help="How much to lower the legend")
    parser.add_argument("--h_pad", type=float, default=-1.0, help="How much to pad the subplots")
    main(parser.parse_args())
