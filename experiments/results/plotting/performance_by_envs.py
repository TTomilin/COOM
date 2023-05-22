from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds = args.seeds
    sequence = args.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    metric = None
    figsize = (8, 8) if n_envs == 4 else (10, 13)
    y_label_shift = -0.06 if n_envs == 4 else -0.04
    share_y = sequence in ['CD4', 'CD8'] or args.metric == 'success'
    fig, ax = plt.subplots(n_envs, 1, sharex='all', sharey=share_y, figsize=figsize)
    iterations = args.task_length * n_envs
    methods = METHODS if n_envs == 4 else METHODS[:-1]

    for i, env in enumerate(envs):
        for j, method in enumerate(methods):
            metric = args.metric if args.metric else METRICS[env] if env in METRICS else 'kills'
            data = get_data(env, iterations, method, metric, seeds, sequence)
            plot_curve(ax[i], args.confidence, PLOT_COLORS[j], TRANSLATIONS[method], iterations, data, len(seeds))

        ax[i].set_ylabel(TRANSLATIONS[metric])
        ax[i].set_title(TRANSLATIONS[env])
        ax[i].yaxis.set_label_coords(y_label_shift, 0.5)

    add_coloured_task_labels(ax[0], envs, sequence, iterations, n_envs)

    fontsize = 9 if n_envs == 4 else 12
    legend_anchor = -0.7 if n_envs == 4 else -1
    plot_name = f'env/{sequence}_{metric}'
    plot_and_save(ax=ax[-1], plot_name=plot_name, n_col=len(methods), legend_anchor=legend_anchor, fontsize=fontsize)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
