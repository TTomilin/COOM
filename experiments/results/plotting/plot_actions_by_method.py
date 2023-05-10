import os

from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    n_actions = 12
    plot_envs, seeds, methods, sequence = args.test_envs, args.seeds, args.methods, args.sequence
    envs = SEQUENCES[sequence]
    n_methods, n_envs = len(methods), len(envs)
    figsize = (6, 6) if n_envs == 4 else (12, 8) if n_methods > 1 else (12, 5)
    max_steps = -np.inf
    cmap = plt.get_cmap('tab20c')
    iterations = args.task_length * n_envs

    if not plot_envs:
        plot_envs = ['train']

    for env in plot_envs:
        fig, ax = plt.subplots(n_methods, 1, figsize=figsize)
        folder = env if env == 'train' else f'test_{env}'

        for j, method in enumerate(methods):
            seed_data = np.empty((len(seeds), iterations, n_actions))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), '../data', 'actions', sequence, method, folder, f'seed_{seed}.json')
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
            sub_plot = ax if n_methods == 1 else ax[j]
            sub_plot.stackplot(np.arange(iterations), mean.T,
                         labels=[TRANSLATIONS[f'Action {i}'] for i in range(n_actions)],
                         colors=[cmap(i) for i in range(n_actions)])
            sub_plot.tick_params(labelbottom=True)
            sub_plot.set_title(TRANSLATIONS[method])
            sub_plot.set_ylabel("Actions")

        top_plot = ax if n_methods == 1 else ax[0]
        add_task_labels(top_plot, envs, max_steps, n_envs)

        title = env.capitalize() if env == 'train' else TRANSLATIONS[SEQUENCES[sequence][env]]
        fig.suptitle(f'        {sequence} - {title}', fontsize=16)

        bottom_plot = ax if n_methods == 1 else ax[-1]
        bottom_plot.set_xlabel("Timesteps (K)")
        n_cols = 4 if n_envs == 4 else 3

        bottom_adjust = 0.07 if n_envs == 4 else 0.13 if n_methods > 1 else 0.2
        plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
        bottom_plot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=n_cols, fancybox=True, shadow=True)

        file_path = '../plots/actions'
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(f'{file_path}/{sequence}_{title}.png')
        plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
