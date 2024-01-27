from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    n_actions = 12
    plot_envs, seeds, method, sequences = args.test_envs, args.seeds, args.method, args.sequences
    envs = SEQUENCES[sequences[0]]
    n_envs, n_sequences = len(envs), len(sequences)
    figsize = (6, 6) if n_envs == 4 else (12, 8) if n_sequences > 1 else (10, 4)
    max_steps = -np.inf
    iterations = args.task_length * n_envs
    cmap = plt.get_cmap('tab20c')
    ep_time_steps = 1000

    if not plot_envs:
        plot_envs = ['train']

    for env in plot_envs:
        fig, ax = plt.subplots(n_sequences, 1, figsize=figsize)
        folder = env if env == 'train' else f'test_{env}'

        for j, sequence in enumerate(sequences):
            data = get_action_data(folder, iterations, method, n_actions, seeds, sequence)

            # Scale the values to add up to 1000 in each time step
            data = data / np.sum(data, axis=1, keepdims=True) * ep_time_steps

            # Create a percent area stackplot with the values in mean
            sub_plot = ax if n_sequences == 1 else ax[j]
            sub_plot.stackplot(np.arange(iterations), data.T,
                               labels=[TRANSLATIONS[f'Action {i}'] for i in range(n_actions)],
                               colors=[cmap(i) for i in range(n_actions)])
            sub_plot.tick_params(labelbottom=True)
            sub_plot.set_title(sequence)
            sub_plot.set_ylabel("Number of Actions", fontsize=14)
            sub_plot.set_xlim(0, iterations)
            sub_plot.set_ylim(0, ep_time_steps)

        top_plot = ax if n_sequences == 1 else ax[0]
        add_task_labels(top_plot, envs, iterations, n_envs)

        title = env.capitalize() if env == 'train' else TRANSLATIONS[SEQUENCES[sequences[0]][env]]
        fig.suptitle(f'        {TRANSLATIONS[method]} - {title}', fontsize=16)

        bottom_plot = ax if n_sequences == 1 else ax[-1]
        bottom_plot.set_xlabel("Timesteps (K)", fontsize=14)
        n_cols = 4 if n_envs == 4 else 3
        bottom_plot.legend(loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=n_cols, fancybox=True, shadow=True,
                           fontsize=11)

        bottom_adjust = 0.07 if n_envs == 4 else -0.03 if n_sequences > 1 else -0.125
        plt.tight_layout(rect=[0, bottom_adjust, 1, 1])

        file_path = 'plots/actions'
        os.makedirs(file_path, exist_ok=True)
        plt.savefig(f'{file_path}/{method}_{"_".join(sequences)}.pdf')
        plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
