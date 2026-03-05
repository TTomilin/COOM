from results.common import *

METRICS_LABELS = ['Average Performance', 'Forgetting', 'Forward Transfer']


def load_cl_data_per_seed(methods, metric, seeds, sequence, data_folder, task_length, tag=''):
    """Load CL test data per seed.

    Returns cl_data of shape (n_methods, n_seeds, n_envs_test, n_envs_train, task_length).
    """
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    n_seeds = len(seeds)
    n_methods = len(methods)
    iterations = n_envs * task_length
    cl_data = np.full((n_methods, n_seeds, n_envs, n_envs, task_length), np.nan)
    results_dir = Path(__file__).parent.parent.resolve()

    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            for k, seed in enumerate(seeds):
                path = results_dir / data_folder / tag / sequence / method / f'seed_{seed}' / f'{env}_{metric}.json'
                if not os.path.exists(path):
                    print(f'Path {path} does not exist')
                    continue
                with open(path, 'r') as f:
                    data = np.array(json.load(f), dtype=np.float64)
                data = np.pad(data, (0, max(0, iterations - len(data))), constant_values=np.nan)
                cl_data[i, k, j] = np.array_split(data[:iterations], n_envs)

    return cl_data


def compute_performance_per_seed(cl_data):
    """Compute average performance per (method, seed).

    cl_data: (n_methods, n_seeds, n_envs_test, n_envs_train, task_length)
    Returns: (n_methods, n_seeds)
    """
    n_methods, n_seeds = cl_data.shape[:2]
    perf = np.full((n_methods, n_seeds), np.nan)
    data_avg = np.nanmean(cl_data, axis=4)  # (n_methods, n_seeds, n_envs, n_envs)
    for m in range(n_methods):
        for s in range(n_seeds):
            d = data_avg[m, s].copy()
            d = np.triu(d)
            d[d == 0] = np.nan
            perf[m, s] = np.nanmean(d)
    return perf


def compute_forgetting_per_seed(cl_data):
    """Compute average forgetting per (method, seed).

    cl_data: (n_methods, n_seeds, n_envs_test, n_envs_train, task_length)
    Returns: (n_methods, n_seeds)
    """
    n_methods, n_seeds = cl_data.shape[:2]
    forg = np.full((n_methods, n_seeds), np.nan)
    end_data = np.nanmean(cl_data[:, :, :, :, -NUM_FINAL_VALS:], axis=4)  # (n_methods, n_seeds, n_envs, n_envs)
    for m in range(n_methods):
        for s in range(n_seeds):
            diag = np.diagonal(end_data[m, s], axis1=0, axis2=1)  # (n_envs,)
            last_col = end_data[m, s, :, -1]                       # (n_envs,)
            f = diag - last_col
            forg[m, s] = np.nanmean(f[:-1])  # exclude the last task (no forgetting possible)
    return forg


def load_transfer_data_per_seed(methods, metric, seeds, sequence, data_folder, task_length):
    """Load current-task training data per seed for forward transfer computation.

    Returns: (n_seeds, n_methods, n_envs * task_length)
    """
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    n_seeds = len(seeds)
    n_methods = len(methods)
    transfer_data = np.full((n_seeds, n_methods, task_length * n_envs), np.nan)
    results_dir = Path(__file__).parent.parent.resolve()

    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            for k, seed in enumerate(seeds):
                path = results_dir / data_folder / sequence / method / f'seed_{seed}' / f'{env}_{metric}.json'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = j * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                steps = len(data)
                transfer_data[k, i, np.arange(task_start, task_start + steps)] = data

    return transfer_data


def compute_metrics(methods, metric, seeds, sequence, data_folder, task_length, confidence):
    """Compute means and CIs for all three metrics for a single sequence."""
    n_seeds = len(seeds)

    cl_data = load_cl_data_per_seed(methods, metric, seeds, sequence, data_folder, task_length)
    perf_per_seed = compute_performance_per_seed(cl_data)
    forg_per_seed = compute_forgetting_per_seed(cl_data)

    perf_mean = np.nanmean(perf_per_seed, axis=1)
    perf_ci = CRITICAL_VALUES[confidence] * np.nanstd(perf_per_seed, axis=1) / np.sqrt(n_seeds)
    forg_mean = np.nanmean(forg_per_seed, axis=1)
    forg_ci = CRITICAL_VALUES[confidence] * np.nanstd(forg_per_seed, axis=1) / np.sqrt(n_seeds)

    transfer_data = load_transfer_data_per_seed(methods, metric, seeds, sequence, data_folder, task_length)
    baseline_data = load_rl_baseline_data(sequence, seeds, task_length, data_folder, metric)
    ft_mean, ft_ci = calculate_transfer(transfer_data, baseline_data, n_seeds, confidence)

    means = np.stack([perf_mean, forg_mean, ft_mean], axis=1)  # (n_methods, 3)
    cis = np.stack([perf_ci, forg_ci, ft_ci], axis=1)          # (n_methods, 3)
    return means, cis


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn-v0_8-deep')
    seeds, metric, sequences, methods = args.seeds, args.metric, args.sequences, args.methods
    n_methods = len(methods)
    n_sequences = len(sequences)

    # Compute metrics for each sequence
    seq_means, seq_cis = [], []
    for sequence in sequences:
        means, cis = compute_metrics(methods, metric, seeds, sequence, args.data_folder,
                                     args.task_length, args.confidence)
        seq_means.append(means)
        seq_cis.append(cis)

    # Layout: n_sequences sub-groups of n_methods bars per subplot
    bar_width = 0.07
    subgroup_gap = 0.12
    subgroup_width = n_methods * bar_width
    group_width = n_sequences * subgroup_width + (n_sequences - 1) * subgroup_gap

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=False)

    for g, (ax, metric_label) in enumerate(zip(axes, METRICS_LABELS)):
        seq_tick_positions = []
        for s, sequence in enumerate(sequences):
            subgroup_start = s * (subgroup_width + subgroup_gap)
            for i, method in enumerate(methods):
                x = subgroup_start + (i + 0.5) * bar_width
                color = METHOD_COLORS.get(method, PLOT_COLORS[i])
                label = TRANSLATIONS[method] if g == 0 and s == 0 else None
                ax.bar(x, seq_means[s][i, g], bar_width, label=label, color=color,
                       yerr=seq_cis[s][i, g], capsize=3,
                       error_kw={'elinewidth': 1.2, 'capthick': 1.2}, zorder=3)
            seq_tick_positions.append(subgroup_start + subgroup_width / 2)

        ax.set_title(metric_label, fontsize=12)
        ax.set_xticks(seq_tick_positions)
        ax.set_xticklabels([TRANSLATIONS[seq] for seq in sequences], fontsize=12)
        ax.tick_params(axis='x', length=0)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, zorder=2)
        ax.grid(True, axis='y', zorder=0)
        ax.grid(False, axis='x')
        ax.set_xlim(-0.15, group_width + 0.15)

    axes[0].set_ylabel('Score', fontsize=11)

    plot_name = f'metrics_histogram_{"_".join(sequences)}'
    n_col = min(n_methods, 5)
    save_and_show(ax=axes[0], plot_name=plot_name, fig=fig, n_col=n_col,
                  vertical_anchor=args.vertical_anchor, bottom_adjust=args.bottom_adjust,
                  add_xlabel=False)


if __name__ == '__main__':
    parser = common_plot_args()
    parser.set_defaults(vertical_anchor=-0.0, bottom_adjust=0.15)
    main(parser.parse_args())
