import os

from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, sequences = cfg.seeds, cfg.metric, cfg.sequences
    n_sequences, n_seeds = len(sequences), len(seeds)
    figsize = (10, 6) if n_sequences > 1 else (9, 3)
    fig, ax = plt.subplots(n_sequences, 1, sharey='all', sharex='all', figsize=figsize)
    colors = COLORS[cfg.sequence]

    for i, sequence in enumerate(sequences):
        ax = ax if n_sequences == 1 else ax[i]
        envs = SEQUENCES[sequence]
        n_envs = len(envs)
        methods = METHODS if n_envs == 4 else METHODS[:-1]
        iterations = cfg.task_length * n_envs
        for j, method in enumerate(methods):
            data = get_data_per_env(envs, iterations, method, metric, seeds, sequence)
            plot_curve(ax, cfg.confidence, colors[j], TRANSLATIONS[method], iterations, data, n_seeds * n_envs,
                       agg_axes=(0, 1))

        ax.set_ylabel('Average Success')
        ax.set_xlim([0, iterations])
        ax.set_ylim([0, 1])
        if n_sequences > 1:
            ax.set_title(sequence, fontsize=14)
        add_task_labels(ax, envs, iterations, n_envs)

    bottom_ax = ax if n_sequences == 1 else ax[-1]
    folder = 'success'
    os.makedirs(folder, exist_ok=True)
    plot_name = f'{folder}/{"_".join(sequences)}'
    bottom_adjust = 0 if n_sequences > 1 else -0.1
    plot_and_save(bottom_ax, plot_name, n_col=len(methods[0]), legend_anchor=-0.5, bottom_adjust=bottom_adjust)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
