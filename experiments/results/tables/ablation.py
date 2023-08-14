import pandas as pd
from typing import Callable

from experiments.results.common import *
from experiments.results.common import calculate_transfer


def main(cfg: argparse.Namespace) -> None:
    methods, seeds, folders, sequence, metric, task_length, confidence = \
        cfg.methods, cfg.seeds, cfg.folders, cfg.sequence, cfg.metric, cfg.task_length, cfg.confidence
    envs = SEQUENCES[sequence]
    n_envs, n_seeds, n_methods, n_folders = len(envs), len(seeds), len(methods), len(folders)

    data = np.empty((n_folders, len(METHODS), 3))
    data_cis = np.empty((n_folders, len(METHODS), 3))
    data[:] = np.nan
    data_cis[:] = np.nan

    for i, folder in enumerate(folders):
        cl_data, ci_data, transfer_data = get_cl_data(methods, metric, seeds, sequence, task_length, confidence,
                                                      folder=folder)
        baseline_data = get_baseline_data(sequence, seeds, task_length, metric)
        performance = calculate_performance(cl_data)
        performance_ci = calculate_performance(ci_data)
        forgetting = calculate_forgetting(cl_data)
        forgetting_ci = calculate_forgetting(ci_data)
        transfer, transfer_ci = calculate_transfer(transfer_data, baseline_data, len(seeds), confidence)

        for j in range(len(methods)):
            data[i, j] = [performance[j], forgetting[j], transfer[j]]
            data_cis[i, j] = [performance_ci[j], forgetting_ci[j], transfer_ci[j]]

    print('Printing ablation study table\n')
    print_table(folders, methods, data, data_cis, value_cell)
    print('\nPrinting ablations result difference table\n')
    print_table(folders, methods, data, data_cis, diff_cell)


def print_table(folders: List[str], methods: List[str], data: ndarray, data_cis: ndarray, get_cell_func: Callable):
    data_types = data.shape[-1]
    results = pd.DataFrame(
        columns=['Method'] + [f'\multicolumn{{{data_types}}}{{c}}{{{TRANSLATIONS[folder]}}}' for folder in folders])
    for i, method in enumerate(methods):
        row = [TRANSLATIONS[method]]
        defaults = data[0, i]
        for j in range(len(folders)):
            cell_values = []
            for k in range(data_types):
                value = data[j, i, k]
                ci = data_cis[j, i, k]
                cell_string = get_cell_func(value, ci, j, k, defaults[k])
                cell_values.append(cell_string)
            cell = ' & '.join(cell_values)
            row.append(cell)
        results.loc[len(results)] = row
    results = results.set_index('Method')
    multi_col_format = 'c@{\hskip 0.05in}c@{\hskip 0.05in}c'
    pd.set_option('display.max_colwidth', None)
    latex_table = results.to_latex(escape=False, column_format='l' + multi_col_format * (len(folders)))
    print(latex_table)


def value_cell(value: float, ci: float, *args) -> str:
    return f'{value:.2f} \tiny ± {ci:.2f}' if not np.isnan(value) else '-'


def diff_cell(value: float, _: float, j: int, k: int, default: ndarray) -> str:
    diff_string = ((value - default) / default) * 100 if k == 0 else value - default
    return f'{value:.2f}' if j == 0 else f' {diff_string:+.2f}' + (r'\%' if k == 0 else '')


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--folders", type=str, required=True, nargs='+', help="Names of the folders")
    main(parser.parse_args())
