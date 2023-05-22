from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    n_actions = 12
    seeds, method, sequence = args.seeds, args.method, args.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    cmap = plt.get_cmap('tab20c')
    iterations = args.task_length * n_envs
    timesteps = 1000

    data = get_action_data('train', iterations, method, n_actions, seeds, sequence)

    # Sum the actions over all the time steps and round to the nearest integer
    total_actions = np.round(np.sum(data, axis=0)).astype(int)

    # Scale the values of total_actions to add up to 1000 in each bin
    total_actions = total_actions / np.sum(total_actions) * timesteps

    fig = plt.figure(figsize=(13, 5))
    y_label = "Number of Actions"

    # Define the grid and specify the width ratios
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

    # STACKPLOT
    stackplot = fig.add_subplot(gs[0])
    stackplot.stackplot(np.arange(iterations), data.T,
                        labels=[TRANSLATIONS[f'Action {i}'] for i in range(n_actions)],
                        colors=[cmap(i) for i in range(n_actions)])
    stackplot.set_ylabel(y_label)
    stackplot.set_xlim(0, iterations)
    stackplot.set_ylim(0, timesteps)

    add_task_labels(stackplot, envs, iterations, n_envs)

    stackplot.set_xlabel("Timesteps (K)", fontsize=11)
    n_cols = 4 if n_envs == 4 else 3

    bottom_adjust = 0.175
    plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
    handles, labels = stackplot.get_legend_handles_labels()

    # HISTOGRAM
    histogram = fig.add_subplot(gs[1])
    histogram.bar(np.arange(n_actions), total_actions, color=[cmap(i) for i in range(n_actions)])
    histogram.set_xticks(np.arange(n_actions))
    histogram.set_xticklabels([TRANSLATIONS[f'Action {i}'] for i in range(n_actions)])
    histogram.set_ylabel(y_label)
    histogram.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Add the legend
    main_ax = fig.add_subplot(1, 1, 1, frameon=False)
    main_ax.axis('off')
    main_ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=n_cols, fancybox=True,
                   shadow=True)

    file_path = 'plots/actions'
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(f'{file_path}/{sequence}_{method}.png')
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
