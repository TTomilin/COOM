from results.common import *


def load_performance_per_method(method, metric, seeds, sequence, data_folder, task_length, confidence, tag):
    cl_data, _, _ = load_cl_data([method], metric, seeds, sequence, data_folder, task_length, confidence, tag=tag)
    return calculate_performance(cl_data)[0]


def main(args: argparse.Namespace) -> None:
    methods, seeds, tags, sequence, metric, task_length, confidence = \
        args.methods, args.seeds, args.tags, args.sequence, args.metric, args.task_length, args.confidence
    ablation_tags = [t for t in tags if t != 'default']
    n_ablations = len(ablation_tags)

    sequences_to_try = [sequence]
    if args.backup_sequence and args.backup_sequence != sequence:
        sequences_to_try.append(args.backup_sequence)

    # Load default and ablation performance for every (method, sequence) combination upfront.
    # Using a dict avoids redundant file reads and makes the diff logic below explicit.
    default_perf = {}   # (method, seq) -> float
    ablation_perf = {}  # (method, seq, tag) -> float
    for method in methods:
        for seq in sequences_to_try:
            perf = load_performance_per_method(method, metric, seeds, seq, args.data_folder,
                                               task_length, confidence, tag='default')
            if not np.isnan(perf):
                default_perf[(method, seq)] = perf
            for tag in ablation_tags:
                perf = load_performance_per_method(method, metric, seeds, seq, args.data_folder,
                                                   task_length, confidence, tag=tag)
                if not np.isnan(perf):
                    ablation_perf[(method, seq, tag)] = perf

    # For each (method, ablation) compute the percentage diff per sequence independently,
    # then average across sequences. This avoids mixing results from different scenario lengths.
    diffs = np.full((len(methods), n_ablations), np.nan)
    for j, method in enumerate(methods):
        for i, tag in enumerate(ablation_tags):
            seq_diffs = []
            for seq in sequences_to_try:
                if (method, seq) in default_perf and (method, seq, tag) in ablation_perf:
                    d = default_perf[(method, seq)]
                    a = ablation_perf[(method, seq, tag)]
                    seq_diffs.append((a - d) / d * 100)
            if seq_diffs:
                diffs[j, i] = np.mean(seq_diffs)
            else:
                print(f'Warning: no valid data for method {method}, tag {tag} in any sequence.')

    plot_histograms(diffs, tags, methods, args.two_rows)


def plot_histograms(diffs: ndarray, ablations: List[str], methods: List[str],
                   two_rows: bool = False):
    plt.style.use('seaborn-v0_8-deep')
    ablations = ablations[1:]  # Remove the original data folder
    n_ablations = len(ablations)

    if two_rows:
        n_cols = int(np.ceil(n_ablations / 2))
        n_rows = 2
    else:
        n_cols = n_ablations
        n_rows = 1

    figsize = (10, 2 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, sharey='all', figsize=figsize)
    axes_flat = axes.flatten() if two_rows else axes

    for i, variation in enumerate(ablations):
        for j, method in enumerate(methods):
            axes_flat[i].bar(i + j, [diffs[j, i]], label=TRANSLATIONS[method], color=METHOD_COLORS[method])

        axes_flat[i].axhline(0, color='black', lw=1)
        axes_flat[i].set_title(TRANSLATIONS[variation], fontsize=11)
        axes_flat[i].set_xticks([])

    # Hide any unused subplots (when n_ablations is odd and two_rows=True)
    for i in range(n_ablations, n_rows * n_cols):
        axes_flat[i].set_visible(False)

    def format_percent(x, pos):
        return f"{x:.0f}%"

    first_ax = axes_flat[0]
    first_ax.yaxis.set_major_formatter(plt.FuncFormatter(format_percent))
    first_ax.set_ylabel('Performance Increase')
    if two_rows:
        axes_flat[n_cols].set_ylabel('Performance Increase')
    plt.ylim(-100, 100)
    save_and_show(axes_flat[n_ablations - 1], plot_name='variations', n_col=len(methods),
                  bottom_adjust=0.1, fig=fig, add_xlabel=False)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--tags", type=str, required=True, nargs='+', help="Names of the wandb tags")
    parser.add_argument("--two_rows", action='store_true', default=False,
                        help="Distribute subplots across 2 rows instead of 1")
    parser.add_argument("--backup_sequence", type=str, default='CO8',
                        choices=['CD4', 'CO4', 'CD8', 'CO8', 'CD16', 'CO16', 'COC'],
                        help="Fallback sequence to use when no data is found for the main sequence")
    main(parser.parse_args())
