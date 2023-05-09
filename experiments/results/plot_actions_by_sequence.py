import os

from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    n_actions = 12
    plot_envs, seeds, method, sequences = args.test_envs, args.seeds, args.method, args.sequences
    envs = SEQUENCES[sequences[0]]
    n_envs = len(envs)
    figsize = (6, 6) if n_envs == 4 else (12, 8)
    max_steps = -np.inf
    iterations = args.task_length * n_envs
    cmap = plt.get_cmap('tab20c')

    if not plot_envs:
        plot_envs = ['train']

    for env in plot_envs:
        fig, ax = plt.subplots(len(sequences), 1, figsize=figsize)
        folder = env if env == 'train' else f'test_{env}'

        for j, sequence in enumerate(sequences):
            seed_data = np.empty((len(seeds), iterations, n_actions))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), 'data', 'actions', sequence, method, folder, f'seed_{seed}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                steps = len(data)
                max_steps = max(max_steps, steps)
                seed_data[k, np.arange(steps)] = data

            # Find the mean actions over all the seeds
            mean = np.nanmean(seed_data, axis=0)
            mean = gaussian_filter1d(mean, sigma=5, axis=0)

            # Scale the values to add up to 1000 in each time step
            mean = mean / np.sum(mean, axis=1, keepdims=True) * 1000

            # Create a percent area stackplot with the values in mean
            ax[j].stackplot(np.arange(iterations), mean.T,
                            labels=[TRANSLATIONS[f'Action {i}'] for i in range(n_actions)],
                            colors=[cmap(i) for i in range(n_actions)])
            ax[j].tick_params(labelbottom=True)
            ax[j].set_title(sequence)
            ax[j].set_ylabel("Actions")

        env_steps = max_steps // n_envs
        task_indicators = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

        tick_labels = [TRANSLATIONS[env] for env in envs]
        ax2 = ax[0].twiny()
        ax2.set_xlim(ax[0].get_xlim())
        ax2.set_xticks(task_indicators)
        ax2.set_xticklabels(tick_labels)
        ax2.tick_params(axis='both', which='both', length=0)

        ax[-1].set_xlabel("Timesteps (K)")
        handles, labels = ax[-1].get_legend_handles_labels()
        n_cols = 4 if n_envs == 4 else 3
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=n_cols, fancybox=True, shadow=True,
                   fontsize=11)

        title = env.capitalize() if env == 'train' else TRANSLATIONS[SEQUENCES[sequences[0]][env]]
        fig.suptitle(f'        {TRANSLATIONS[method]} - {title}', fontsize=16)
        bottom_adjust = 0.07 if n_envs == 4 else 0.13
        plt.tight_layout(rect=[0, bottom_adjust, 1, 1])

        file_path = 'plots/actions'
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(f'{file_path}/{method}_{title}.png')
        plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--sequences", type=str, default=['CO8', 'COC'], choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'],
                        help="Name of the task sequence")
    main(parser.parse_args())
