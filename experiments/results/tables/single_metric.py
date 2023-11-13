import pandas as pd

from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    methods, seeds, sequences, task_length, confidence = \
        cfg.methods, cfg.seeds, cfg.sequences, cfg.task_length, cfg.confidence

    methods = METHODS if methods is None else methods

    data = np.full((len(sequences), len(METHODS)), np.nan)
    data_cis = np.full((len(sequences), len(METHODS)), np.nan)

    for i, sequence in enumerate(sequences):
        cl_data, ci_data, transfer_data = get_cl_data(methods, cfg.metric, seeds, sequence, task_length, confidence)
        baseline_data = get_baseline_data(sequence, seeds, cfg.task_length, cfg.metric)
        metric_data, metric_data_ci = calculate_metric_data(cfg.cl_metric, cl_data, ci_data, transfer_data, baseline_data,
                                                            seeds, confidence)

        for j in range(len(methods)):
            data[i, j] = metric_data[j]
            data_cis[i, j] = metric_data_ci[j]

    print_combined(methods, sequences, data, data_cis, cfg.cl_metric)


def calculate_metric_data(metric, cl_data, ci_data, transfer_data, baseline_data, seeds, confidence):
    if metric == 'success':
        metric_data = calculate_performance(cl_data)
        metric_data_ci = calculate_performance(ci_data)
    elif metric == 'forgetting':
        metric_data, _ = calculate_forgetting(cl_data)
        metric_data_ci, _ = calculate_forgetting(ci_data)
    elif metric == 'transfer':
        metric_data, metric_data_ci = calculate_transfer(transfer_data, baseline_data, len(seeds), confidence)
    else:
        raise ValueError(f'Unknown metric {metric}')
    return metric_data, metric_data_ci


def print_combined(methods, sequences, data, data_cis, metric):
    pd.set_option('display.max_colwidth', None)
    results = pd.DataFrame(columns=['Method'] + sequences + ['Average'])
    highlight_func = np.nanmin if metric == 'forgetting' else np.nanmax

    for i, method in enumerate(methods):
        row = [TRANSLATIONS[method]]
        for j, sequence in enumerate(sequences):
            value = data[j, i]
            ci = data_cis[j, i]
            significant = highlight_func(data[j]) == value
            cell_string = f'\\textbf{{{value:.2f}}} \tiny ± {ci:.2f}' if significant else f'{value:.2f} \tiny ± {ci:.2f}' \
                if not np.isnan(value) else '-'
            cell_string = cell_string.replace('-0.00', '0.00')
            row.append(cell_string)
        value = np.nanmean(data[:, i])
        ci = np.nanmean(data_cis[:, i])
        significant = highlight_func(np.nanmean(data, axis=0)) == value
        avg_string = f'\\textbf{{{value:.2f}}} \tiny ± {ci:.2f}' if significant else f'{value:.2f} \tiny ± {ci:.2f}' \
            if not np.isnan(value) else '-'
        avg_string = avg_string.replace('-0.00', '0.00')
        row.append(avg_string)
        results.loc[len(results)] = row
    results = results.set_index('Method')
    latex_table = results.to_latex(escape=False, column_format='l' + 'c' * (len(sequences) + 1))
    print(latex_table)


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--cl_metric", type=str, default='success', help="Name of the cl metric to store/plot")
    parser.add_argument("--second_half", default=False, action='store_true', help="Only regard sequence 2nd half")
    parser.add_argument("--task_forgetting", default=False, action='store_true', help="Only print task forgetting")
    main(parser.parse_args())
