from results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds, metric, sequence = cfg.seeds, cfg.metric, cfg.sequence
    colors = COLORS[sequence]
    envs = SEQUENCES[sequence]
    n_seeds, n_envs = len(seeds), len(envs)
    methods = cfg.methods if cfg.methods else METHODS if n_envs == 4 else METHODS[:-1]
    n_methods = len(methods)
    figsize = (12, 14) if n_methods > 1 else (11, 2.25)
    fig, ax = plt.subplots(len(methods), 1, sharey='all', sharex='all', figsize=figsize)
    n_data_points = cfg.task_length * n_envs
    iterations = n_data_points * LOG_INTERVAL

    for i, method in enumerate(methods):
        cur_ax = ax if n_methods == 1 else ax[i]
        for j, env in enumerate(envs):
            data = get_data(env, n_data_points, method, metric, seeds, sequence)
            plot_curve(cur_ax, cfg.confidence, colors[j], TRANSLATIONS[env], iterations, data, n_seeds,
                       interval=LOG_INTERVAL, sigma=4)
        if n_methods > 1:
            cur_ax.set_title(TRANSLATIONS[method], fontsize=12)
        cur_ax.set_ylabel(TRANSLATIONS[metric])
        cur_ax.set_xlim([0, iterations])
        cur_ax.set_ylim([0, 1])

    top_ax = ax if n_methods == 1 else ax[0]
    bottom_ax = ax if n_methods == 1 else ax[-1]
    add_coloured_task_labels(top_ax, sequence, iterations, fontsize=9)
    n_cols = n_envs if n_envs == 4 else n_envs // 2 if n_methods > 1 else 1
    method = f'_{methods[0]}' if n_methods == 1 else ''
    plot_name = f'method/{sequence}_{metric}{method}'
    vertical_anchor = -0.8 if n_methods > 1 else -0.1
    bottom_adjust = 0 if n_methods > 1 else 0
    loc = 'lower center' if n_methods > 1 else 'lower right'
    horizontal_anchor = 1.225 if n_methods == 1 else 0.5
    plot_and_save(ax=bottom_ax, plot_name=plot_name, n_col=n_cols, vertical_anchor=vertical_anchor,
                  bottom_adjust=bottom_adjust, loc=loc, horizontal_anchor=horizontal_anchor, add_xlabel=n_methods > 1,
                  add_legend=False)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
