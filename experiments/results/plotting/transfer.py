import os

from experiments.results.common import *
from experiments.results.common import add_main_ax

COLORS = ['#C44E52', '#55A868']


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    seeds = args.seeds
    sequences = ['single', args.sequence]
    scenarios = SEQUENCES[args.sequence]
    n_envs = len(scenarios)
    metric = None
    n_rows = 2
    n_cols = int(np.ceil(n_envs / n_rows))
    figsize = (5, 4) if n_envs == 4 else (11, 5)
    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', figsize=figsize)
    task_length = args.task_length

    for i, env in enumerate(scenarios):
        row = i % n_cols
        col = i // n_cols
        reference = None
        for j, sequence in enumerate(sequences):
            method = 'sac' if sequence == 'single' else args.method
            metric = METRICS[env] if args.metric == 'env' else args.metric
            seed_data = np.empty((len(seeds), task_length))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                data_file = env if sequence == 'single' else 'train'
                path = os.path.join(os.getcwd(), 'data', sequence, method, f'seed_{seed}', f'{data_file}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = 0 if sequence == 'single' else i * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                steps = len(data)
                seed_data[k, np.arange(steps)] = data

            mean = np.nanmean(seed_data, axis=0)
            mean = gaussian_filter1d(mean, sigma=2)

            ax[col, row].plot(mean, label=TRANSLATIONS[method], color=COLORS[j])
            ax[col, row].tick_params(labelbottom=True)
            ax[col, row].ticklabel_format(style='sci', axis='y', scilimits=(0, 4))
            if reference is None:
                reference = mean
            else:
                ax[col, row].fill_between(np.arange(task_length), mean, reference, where=(mean < reference), alpha=0.2,
                                          color=COLORS[0], interpolate=True)
                ax[col, row].fill_between(np.arange(task_length), mean, reference, where=(mean >= reference), alpha=0.2,
                                          color=COLORS[1], interpolate=True)

        ax[col, row].set_ylabel(TRANSLATIONS[metric], fontsize=9)
        ax[col, row].set_title(TRANSLATIONS[env], fontsize=11)
        ax[col, row].yaxis.set_label_coords(-0.27, 0.5)

    add_main_ax(fig)

    handles, labels = ax[-1, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fancybox=True, shadow=True)
    fig.tight_layout()
    plt.savefig(f'plots/transfer/{args.sequence}_{args.method}_individual.png')
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
