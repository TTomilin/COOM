from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    methods, seeds, metric, sequences = cfg.methods, cfg.seeds, cfg.metric, cfg.sequences
    n_sequences, n_seeds = len(sequences), len(seeds)
    figsize = (12, 7) if n_sequences > 1 else (9, 3)
    fig, axes = plt.subplots(n_sequences, 1, sharey='all', sharex='all', figsize=figsize)
    assert len(sequences) > 0, "No sequences provided"

    for i, sequence in enumerate(sequences):
        ax = axes if n_sequences == 1 else axes[i]
        envs = SEQUENCES[sequence]
        n_envs = len(envs)
        if methods is None:
            methods = METHODS if n_envs == 4 else METHODS[:-1]
        iterations = cfg.task_length * n_envs * LOG_INTERVAL
        n_data_points = cfg.task_length * n_envs
        for j, method in enumerate(methods):
            data = get_data_per_env(envs, n_data_points, method, metric, seeds, sequence)
            plot_curve(ax, cfg.confidence, PLOT_COLORS[j], TRANSLATIONS[method], iterations, data, n_seeds * n_envs,
                       agg_axes=(0, 1), interval=LOG_INTERVAL)

        ax.set_ylabel('Average Success')
        ax.set_xlim([0, iterations])
        ax.set_ylim([0, 1])
        if n_sequences > 1:
            ax.set_title(sequence, fontsize=14)
        add_task_labels(ax, envs, iterations, n_envs, fontsize=9)

    if len(methods) > 1:
        if n_sequences > 1:
            bottom_ax = axes if n_sequences == 1 else axes[-1]
            bottom_ax.set_xlabel("Timesteps (K)", fontsize=10)
            bottom_ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=len(methods), fancybox=True, shadow=True)
        else:
            ax.legend(ncol=2)
    plt.tight_layout()
    folder = 'success'
    os.makedirs(folder, exist_ok=True)
    plot_name = f'plots/{folder}/{"_".join(sequences)}'
    print(f'Saving plot to {plot_name}')
    plt.savefig(f'{plot_name}.png')
    plt.savefig(f'{plot_name}.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
