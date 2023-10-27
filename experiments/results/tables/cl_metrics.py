import pandas as pd

from experiments.results.common import *
from experiments.results.common import calculate_performance, get_cl_data, calculate_forgetting


def print_results(metric_data: np.ndarray, ci: np.ndarray, methods: List[str], metric: str):
    print(f'\n{"Method":<20}{metric:<20}{"CI":<20}')
    for i, method in enumerate(methods):
        print(f'{method:<20}{metric_data[i]:<20.2f}{ci[i]:.2f}')


def normalize(metric_data, ci):
    joined_results = np.array((metric_data, ci))
    joined_results = joined_results / np.linalg.norm(joined_results)
    return joined_results


def main(cfg: argparse.Namespace) -> None:
    methods, seeds, sequences, metric, task_length, confidence = \
        cfg.methods, cfg.seeds, cfg.sequences, cfg.metric, cfg.task_length, cfg.confidence

    # Create 3-dimensional arrays to store performance, forgetting, transfer + their confidence intervals
    data = np.empty((len(sequences), len(METHODS), 3))
    data_cis = np.empty((len(sequences), len(METHODS), 3))
    data[:] = np.nan
    data_cis[:] = np.nan

    for i, sequence in enumerate(sequences):
        cl_data, ci_data, transfer_data = get_cl_data(methods, metric, seeds, sequence, task_length, confidence,
                                                      second_half=cfg.second_half)
        baseline_data = get_baseline_data(sequence, seeds, cfg.task_length, cfg.metric)
        performance = calculate_performance(cl_data)
        performance_ci = calculate_performance(ci_data)
        forgetting, forgetting_individual = calculate_forgetting(cl_data)
        forgetting_ci, forgetting_individual_ci = calculate_forgetting(ci_data)
        transfer, transfer_ci = calculate_transfer(transfer_data, baseline_data, len(seeds), confidence)

        for j in range(len(methods)):
            data[i, j] = [performance[j], forgetting[j], transfer[j]]
            data_cis[i, j] = [performance_ci[j], forgetting_ci[j], transfer_ci[j]]

    if cfg.task_forgetting:
        print_task_forgetting(methods, sequence, forgetting_individual, forgetting_individual_ci)
    else:
        print_combined(methods, sequences, data, data_cis)


def print_task_forgetting(methods: List[str], sequence: str, forgetting: ndarray, cis: ndarray):
    pd.set_option('display.max_colwidth', None)
    envs = SEQUENCES[sequence]
    method_names = [TRANSLATIONS[method] for method in methods]
    results = pd.DataFrame(columns=['Scenario'] + method_names + ['Average'])
    for i, env in enumerate(envs):
        env_name = TRANSLATIONS[env]
        row = [env_name] + ['-' if np.isnan(forgetting[j, i]) or i == len(envs) - 1 else f'{forgetting[j, i]:.2f} \tiny ± {cis[j, i]:.2f}' for j in range(len(methods))]
        method_forget = [forgetting[j, i] for j in range(len(methods))]
        row += [f'{np.nanmean(method_forget):.2f} \tiny ± {np.nanstd(method_forget):.2f}']
        results.loc[len(results)] = row

    def highlight_max(s):
        s_arr = s.to_numpy()  # convert series to numpy array
        is_max = s_arr == s_arr.max()
        return ['\\textbf{' + str(val) + '}' if is_max[idx] and not val == '-' else str(val) for idx, val in
                enumerate(s_arr)]

    results = results.set_index('Scenario')
    results = results.apply(highlight_max, axis=0)
    latex_table = results.to_latex(escape=False, column_format='l' + 'c' * len(methods))  # Adjust column format
    print(latex_table)


def print_combined(methods: List[str], sequences, data, data_cis):
    data_types = data.shape[-1]
    pd.set_option('display.max_colwidth', None)
    results = pd.DataFrame(
        columns=['Method'] + [f'\multicolumn{{{data_types}}}{{c}}{{{sequence}}}' for sequence in sequences] + [
            '\multicolumn{3}{c}{Average}'])
    highlight_func = [np.nanmax if k in [0, 2] else np.nanmin for k in range(3)]
    methods = METHODS if methods is None else methods
    for i, method in enumerate(methods):
        row = [TRANSLATIONS[method]]
        for j, sequence in enumerate(sequences):
            cell_values = []
            for k in range(data_types):
                value = data[j, i, k]
                ci = data_cis[j, i, k]
                significant = highlight_func[k](data[j, :, k]) == value
                cell_string = f'\\textbf{{{value:.2f}}} ± {ci:.2f}' if significant else f'{value:.2f} ± {ci:.2f}' \
                    if not np.isnan(value) else '-'
                cell_values.append(cell_string)
            cell = ' & '.join(cell_values)
            row.append(cell)
        cell_values = []
        for k in range(data_types):
            value = np.nanmean(data[:, i, k])
            ci = np.nanmean(data_cis[:, i, k])
            significant = highlight_func[k](np.nanmean(data[:, :, k], axis=0)) == value
            avg_string = f'\\textbf{{{value:.2f}}} ± {ci:.2f}' if significant else f'{value:.2f} ± {ci:.2f}' \
                if not np.isnan(value) else '-'
            cell_values.append(avg_string)
        cell = ' & '.join(cell_values)
        row.append(cell)
        results.loc[len(results)] = row
    results = results.set_index('Method')
    multi_col_format = 'c@{\hskip 0.05in}c@{\hskip 0.05in}c'
    latex_table = results.to_latex(escape=False, column_format='l' + multi_col_format * (len(sequences) + 1))
    print(latex_table)


def print_latex(sequences, mean, ci, highlight_max=True):
    results = pd.DataFrame(columns=['algorithm'] + sequences + ['Average'])
    for i, method in enumerate(METHODS):
        row = [TRANSLATIONS[method]] + [f'{mean[j][i]:.2f} \tiny ± {ci[j][i]:.2f}' if not np.isnan(mean[j][i]) else '-'
                                        for j in range(len(sequences))]
        avg = np.nanmean([mean[j][i] for j in range(len(sequences))])
        row += [f'{avg:.2f} \tiny ± {np.nanstd([mean[j][i] for j in range(len(sequences))]):.2f}']
        results.loc[len(results)] = row
    results = results.set_index('algorithm')

    def highlight_max(s):
        s_arr = s.to_numpy()  # convert series to numpy array
        is_max = s_arr == s_arr.max()
        return ['\\textbf{' + str(val) + '}' if is_max[idx] and not val == '-' else str(val) for idx, val in
                enumerate(s_arr)]

    results = results.apply(highlight_max, axis=0)
    print(results.to_latex(escape=False))


def print_latex_swapped(sequences, mean, ci, highlight_max=True):
    methods = [TRANSLATIONS[method] for method in METHODS]
    results = pd.DataFrame(columns=['sequence'] + methods)
    for i, seq in enumerate(sequences):
        row = [seq] + [f'{mean[i][j]:.2f} \tiny ± {ci[i][j]:.2f}' if not np.isnan(mean[i][j]) else '-'
                       for j in range(len(METHODS))]
        index = np.nanargmax(mean[i]) if highlight_max else np.nanargmin(mean[i])
        row[index + 1] = f'\textbf{{{row[index + 1]}}}'
        results.loc[len(results)] = row

    average_row = ['Average'] + [f'{np.nanmean(mean[:, j]):.2f} \tiny ± {np.nanmean(ci[:, j]):.2f}'
                                 for j in range(len(METHODS))]
    results.loc[len(results)] = average_row

    results = results.set_index('sequence')
    print(results.to_latex(escape=False))


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--second_half", default=False, action='store_true', help="Only regard sequence 2nd half")
    parser.add_argument("--task_forgetting", default=False, action='store_true', help="Only print task forgetting")
    main(parser.parse_args())
