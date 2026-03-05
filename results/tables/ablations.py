import pandas as pd

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

    # Pre-compute diffs per sequence independently, then average — matches plotting/ablations.py exactly.
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

    # Baseline performance per method: first sequence that has data.
    default_values = np.full(len(methods), np.nan)
    for j, method in enumerate(methods):
        for seq in sequences_to_try:
            if (method, seq) in default_perf:
                default_values[j] = default_perf[(method, seq)]
                break

    print('Printing ablation study table\n')
    print_ablation_table(ablation_tags, methods, default_values, diffs)


def print_ablation_table(ablation_tags: List[str], methods: List[str], default_values: ndarray,
                         diffs: ndarray) -> None:
    columns = ['Method', 'Default'] + [TRANSLATIONS[tag] for tag in ablation_tags]
    results = pd.DataFrame(columns=columns)
    for j, method in enumerate(methods):
        row = [TRANSLATIONS[method], f'{default_values[j]:.2f}' if not np.isnan(default_values[j]) else '-']
        for i in range(len(ablation_tags)):
            diff = diffs[j, i]
            row.append(f'{diff:+.2f}\\%' if not np.isnan(diff) else '-')
        results.loc[len(results)] = row
    results = results.set_index('Method')
    pd.set_option('display.max_colwidth', None)
    n_cols = 1 + len(ablation_tags)
    print(results.to_latex(escape=False, column_format='l' + 'c' * n_cols))



if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--tags", type=str, required=True, nargs='+', help="Names of the wandb tags")
    parser.add_argument("--backup_sequence", type=str, default='CO8',
                        choices=['CD4', 'CO4', 'CD8', 'CO8', 'CD16', 'CO16', 'COC'],
                        help="Fallback sequence to use when no data is found for the main sequence")
    main(parser.parse_args())
