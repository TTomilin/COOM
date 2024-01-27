from results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    n_actions = 12
    seeds, method, sequence = args.seeds, args.method, args.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    fig, ax = plt.subplots(n_envs, 1, figsize=(11, 18))
    cmap = plt.get_cmap('tab20c')
    iterations = args.task_length * n_envs
    ep_time_steps = 1000

    for i, env in enumerate(envs):
        folder = f'test_{i}'
        data = get_action_data(folder, iterations, method, n_actions, seeds, sequence)

        # Create a percent area stackplot with the values in mean
        ax[i].stackplot(np.arange(iterations), data.T,
                        labels=[TRANSLATIONS[f'Action {i}'] for i in range(n_actions)],
                        colors=[cmap(i) for i in range(n_actions)])
        ax[i].tick_params(labelbottom=True)
        ax[i].set_title(TRANSLATIONS[env])
        ax[i].set_ylabel("Number of Actions")
        ax[i].set_xlim(0, iterations)
        ax[i].set_ylim(0, ep_time_steps)

    add_task_labels(ax[0], envs, iterations, n_envs)

    ax[-1].set_xlabel("Timesteps (K)", fontsize=11)
    n_cols = 4 if n_envs == 4 else 3

    plt.tight_layout(rect=[0, 0.065, 1, 1])
    ax[-1].legend(loc='lower center', bbox_to_anchor=(0.5, -1.1), ncol=n_cols, fancybox=True, shadow=True)

    file_path = 'plots/actions'
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(f'{file_path}/{sequence}_{method}_all.pdf')
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
