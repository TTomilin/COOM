import os

from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-paper')
    seeds, sequences = args.seeds, args.sequences
    colors = COLORS[sequences[0]]
    envs = SEQUENCES[sequences[0]]
    n_envs = len(envs)
    metric = None
    n_rows = 2
    n_cols = int(np.ceil(n_envs / n_rows))
    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', figsize=(10, 4))
    max_steps = -np.inf
    task_length = args.task_length

    for i, env in enumerate(envs):
        row = i % n_cols
        col = i // n_cols
        for j, sequence in enumerate(sequences):
            for l, method in enumerate(args.methods):
                metric = args.metric if args.metric else METRICS[env]
                seed_data = np.empty((len(seeds), task_length))
                seed_data[:] = np.nan
                for k, seed in enumerate(seeds):
                    path = os.path.join(os.getcwd(), 'data', sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                    if not os.path.exists(path):
                        continue
                    with open(path, 'r') as f:
                        task_start = i * task_length
                        data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    max_steps = max(max_steps, steps)
                    seed_data[k, np.arange(steps)] = data
                label = f'{TRANSLATIONS[method]} ({TRANSLATIONS[sequence]})'
                plot_curve(ax[col, row], args.confidence, colors[l], label, args.task_length, seed_data,
                           len(seeds), linestyle=LINE_STYLES[j])

        ax[col, row].set_ylabel(TRANSLATIONS[metric])
        ax[col, row].set_title(TRANSLATIONS[env])
        ax[col, row].yaxis.set_label_coords(-0.2, 0.5)
        ax[col, row].ticklabel_format(axis='y', style='sci', scilimits=(0, 5))

    add_main_ax(fig)
    handles, labels = ax[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=6, fancybox=True, shadow=True)
    fig.tight_layout()
    plt.savefig('plots/COC/train_per_env.png')
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
