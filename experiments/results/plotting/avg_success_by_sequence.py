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
        iterations = cfg.task_length * n_envs * LOG_INTERVAL
        n_data_points = int(iterations / 1000)
        for j, method in enumerate(methods):
            data = get_data_per_env(envs, n_data_points, method, metric, seeds, sequence)
            plot_curve(ax, cfg.confidence, colors[j], TRANSLATIONS[method], iterations, data, n_seeds * n_envs,
                       agg_axes=(0, 1))

        ax.set_ylabel('Average Success')
        ax.set_xlim([0, iterations])
        ax.set_ylim([0, 1])
        if n_sequences > 1:
            ax.set_title(sequence, fontsize=14)
        add_task_labels(ax, envs, iterations, n_envs, fontsize=9)

    # bottom_ax = ax if n_sequences == 1 else ax[-1]
    # bottom_ax.set_xlabel("Timesteps (K)", fontsize=10)
    # plt.ticklabel_format(axis='x', style='sci', scilimits=(1, 1))
    ax.legend(ncol=2)
    folder = 'success'
    os.makedirs(folder, exist_ok=True)
    plot_name = f'{folder}/{"_".join(sequences)}'
    plt.tight_layout()
    plt.savefig(f'plots/{plot_name}.pdf')
    plt.show()

if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
