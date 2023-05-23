from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, sequences, methods = cfg.seeds, cfg.metric, cfg.sequences, cfg.methods
    n_envs = len(SEQUENCES[sequences[0]])
    fig, ax = plt.subplots(n_envs, 1, sharey='all', sharex='all', figsize=(11, 16))
    iterations = cfg.task_length * n_envs

    for i, sequence in enumerate(sequences):
        envs = SEQUENCES[sequence][:n_envs]
        colors = COLORS[sequence]
        for j, env in enumerate(envs):
            for k, method in enumerate(methods):
                data = get_data(env, iterations, method, metric, seeds, sequence)
                plot_curve(ax[j], cfg.confidence, colors[k], f'{TRANSLATIONS[method]} ({sequence})', iterations, data, len(seeds),
                           linestyle=LINE_STYLES[i])

            ax[j].set_ylabel(TRANSLATIONS[metric])
            ax[j].set_title(TRANSLATIONS[env])
            ax[j].set_xlim([0, iterations])
            ax[j].set_ylim([0, 1])
    plot_name = f'forgetting/{"vs".join(cfg.sequences)}'
    plot_and_save(ax=ax[-1], plot_name=plot_name, n_col=8, legend_anchor=-0.6)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
