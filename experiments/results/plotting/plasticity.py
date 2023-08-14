from experiments.results.common import *


# Logging of some metrics begins when the corresponding task starts
offsets = {
    'CO4': [0, 0, 0, 3],
    'CO8': [0, 1, 0, 0, 4, 0, 4, 2]
}


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    seeds, sequence, n_repeats, task_length = args.seeds, args.sequence, args.n_repeats, args.task_length
    scenarios = SEQUENCES[args.sequence]
    n_envs = len(scenarios)
    n_seeds = len(seeds)
    n_rows = 1 if n_envs == 4 else 2
    n_cols = int(np.ceil(n_envs / n_rows))
    figsize = (10, 3.5) if n_envs == 4 else (11, 6)
    fig, ax = plt.subplots(n_rows, n_cols, sharex='all', figsize=figsize)
    n_data_points = task_length * n_envs * n_repeats
    method = 'fine_tuning'

    for i, env in enumerate(scenarios):
        row = i % n_cols
        col = i // n_cols
        metric = METRICS[env]
        seed_data = np.empty((n_seeds, task_length * n_repeats))
        seed_data[:] = np.nan
        for k, seed in enumerate(seeds):
            path = os.path.join(os.getcwd(), 'data', 'repeat_10', sequence, method, f'seed_{seed}', f'train_{metric}.json')
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                data = np.array(json.load(f))
            task_start = i * task_length - offsets[sequence][i] * task_length
            start_time_steps = np.arange(task_start, n_data_points, n_envs * task_length)
            start_time_steps = start_time_steps[start_time_steps < len(data)]  # In case of early stopping
            data = [data[env_data_start_point: env_data_start_point + task_length] for env_data_start_point in start_time_steps]
            data = np.concatenate(data)  # Concatenate data from all repeats
            steps = len(data)
            seed_data[k, np.arange(steps)] = data

        iterations = task_length * n_repeats * LOG_INTERVAL
        cur_ax = ax[col, row] if n_rows > 1 else ax[row]
        plot_curve(cur_ax, args.confidence, PLOT_COLORS[0], TRANSLATIONS[method], iterations, seed_data, len(seeds),
                   interval=LOG_INTERVAL)

        add_main_ax(fig, fontweight='ultralight', labelpad=30)

        cur_ax.set_ylabel(TRANSLATIONS[metric], fontsize=9)

        cur_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 4), useMathText=True)
        cur_ax.set_title(TRANSLATIONS[env], fontsize=11)

    fig.suptitle(sequence, fontsize=16)
    fig.tight_layout()
    plt.savefig(f'plots/plasticity/{sequence}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--n_repeats", type=int, default=10, help="Number of task sequence repeats")
    main(parser.parse_args())
