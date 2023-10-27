import pandas as pd
from typing import Callable

from experiments.results.common import *
from experiments.results.common import calculate_transfer


def main(cfg: argparse.Namespace) -> None:
    methods, seeds, folders, main_sequence, alt_sequence, metric, task_length, confidence = \
        cfg.methods, cfg.seeds, cfg.folders, cfg.sequence, cfg.alt_sequence, cfg.metric, cfg.task_length, cfg.confidence
    envs = SEQUENCES[main_sequence]
    n_envs, n_seeds, n_methods, n_folders = len(envs), len(seeds), len(methods), len(folders)

    data = np.empty((n_folders, len(METHODS), 3))
    data_cis = np.empty((n_folders, len(METHODS), 3))
    data[:] = np.nan
    data_cis[:] = np.nan

    for i, folder in enumerate(folders):
        cl_data, ci_data, transfer_data = get_cl_data(methods, metric, seeds, main_sequence, task_length, confidence,
                                                      folder=folder)
        cl_data_alt, ci_data_alt, transfer_data_alt = get_cl_data(methods, metric, seeds, alt_sequence, task_length,
                                                                  confidence, folder=folder)
        baseline_data = get_baseline_data(main_sequence, seeds, task_length, metric)
        performance = calculate_performance(cl_data)
        performance_ci = calculate_performance(ci_data)
        forgetting, _ = calculate_forgetting(cl_data)
        forgetting_ci, _ = calculate_forgetting(ci_data)
        transfer, transfer_ci = calculate_transfer(transfer_data, baseline_data, len(seeds), confidence)

        # If performance contains nan values, replace those nan values with alternative data to calculate performance
        if np.any(np.isnan(performance)):
            performance_alt = calculate_performance(cl_data_alt)
            performance_ci_alt = calculate_performance(ci_data_alt)
            forgetting_alt = calculate_forgetting(cl_data_alt)
            forgetting_ci_alt = calculate_forgetting(cl_data_alt)
            transfer_alt, transfer_ci_alt = calculate_transfer(transfer_data_alt, baseline_data, len(seeds), confidence)
            for j in range(len(methods)):
                if np.isnan(performance[j]):
                    performance[j] = performance_alt[j]
                    performance_ci[j] = performance_ci_alt[j]
                    forgetting[j], _ = forgetting_alt[j]
                    forgetting_ci[j], _ = forgetting_ci_alt[j]
                    transfer[j] = transfer_alt[j]
                    transfer_ci[j] = transfer_ci_alt[j]

        for j in range(len(methods)):
            data[i, j] = [performance[j], forgetting[j], transfer[j]]
            data_cis[i, j] = [performance_ci[j], forgetting_ci[j], transfer_ci[j]]

    print('Printing ablation study table\n')
    print_performance(folders, methods, data, data_cis, value_cell)
    print('\nPrinting ablations result difference table\n')
    print_performance(folders, methods, data, data_cis, diff_cell)


def print_performance(folders: List[str], methods: List[str], data: ndarray, data_cis: ndarray, get_cell_func: Callable):
    data = data[:, :, 0]
    data_cis = data_cis[:, :, 0]
    results = pd.DataFrame(columns=['Method'] + [f'{TRANSLATIONS[folder]}' for folder in folders])
    for i, method in enumerate(methods):
        row = [TRANSLATIONS[method]]
        default = data[0, i]
        for j in range(len(folders)):
            value = data[j, i]
            ci = data_cis[j, i]
            cell = get_cell_func(value, ci, j, default=default)
            row.append(cell)
        results.loc[len(results)] = row
    results = results.set_index('Method')
    multi_col_format = 'c@{\hskip 0.05in}c@{\hskip 0.05in}c'
    pd.set_option('display.max_colwidth', None)
    latex_table = results.to_latex(escape=False, column_format='l' + multi_col_format * (len(folders)))
    print(latex_table)


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


def value_cell(value: float, ci: float, *args, **kwargs) -> str:
    return f'{value:.2f} \tiny ± {ci:.2f}' if not np.isnan(value) else '-'


def diff_cell(value: float, _: float, j: int, k: int = 0, default: ndarray = None) -> str:
    diff_string = ((value - default) / default) * 100 if k == 0 else value - default
    return f'{value:.2f}' if j == 0 else f' {diff_string:+.2f}' + (r'\%' if k == 0 else '')


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--folders", type=str, required=True, nargs='+', help="Names of the folders")
    main(parser.parse_args())
