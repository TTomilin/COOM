from experiments.results.common import *
from experiments.results.common import get_action_data


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    n_actions = 12
    plot_envs, seeds, methods, sequence = args.test_envs, args.seeds, args.methods, args.sequence
    envs = SEQUENCES[sequence]
    n_methods, n_envs = len(methods), len(envs)
    figsize = (6, 6) if n_envs == 4 else (12, 8) if n_methods > 1 else (12, 5)
    cmap = plt.get_cmap('tab20c')
    iterations = args.task_length * n_envs
    ep_time_steps = 1000

    if not plot_envs:
        plot_envs = ['train']

    for env in plot_envs:
        fig, ax = plt.subplots(n_methods, 1, figsize=figsize)
        folder = env if env == 'train' else f'test_{env}'

        for j, method in enumerate(methods):
            data = get_action_data(folder, iterations, method, n_actions, seeds, sequence)

            # Create a percent area stackplot with the values in mean
            sub_plot = ax if n_methods == 1 else ax[j]
            sub_plot.stackplot(np.arange(iterations), data.T,
                         labels=[TRANSLATIONS[f'Action {i}'] for i in range(n_actions)],
                         colors=[cmap(i) for i in range(n_actions)])
            sub_plot.tick_params(labelbottom=True)
            sub_plot.set_title(TRANSLATIONS[method])
            sub_plot.set_ylabel("Number of Actions")
            sub_plot.set_xlim(0, iterations)
            sub_plot.set_ylim(0, ep_time_steps)

        top_plot = ax if n_methods == 1 else ax[0]
        add_task_labels(top_plot, envs, iterations, n_envs)

        title = env.capitalize() if env == 'train' else TRANSLATIONS[SEQUENCES[sequence][env]]
        fig.suptitle(f'        {sequence} - {title}', fontsize=16)

        bottom_plot = ax if n_methods == 1 else ax[-1]
        bottom_plot.set_xlabel("Timesteps (K)", fontsize=11)
        n_cols = 4 if n_envs == 4 else 3

        bottom_adjust = 0.07 if n_envs == 4 else 0.13 if n_methods > 1 else 0.2
        plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
        bottom_plot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=n_cols, fancybox=True, shadow=True)

        file_path = 'plots/actions'
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(f'{file_path}/{sequence}_{title}.pdf')
        plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
