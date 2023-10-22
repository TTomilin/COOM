from typing import Callable

import pandas as pd

from experiments.results.common import *
from experiments.results.common import calculate_transfer


def main(cfg: argparse.Namespace) -> None:
    methods, seeds, folders, main_sequence, alt_sequence, metric, task_length, confidence = \
        cfg.methods, cfg.seeds, cfg.folders, cfg.sequence, cfg.alt_sequence, cfg.metric, cfg.task_length, cfg.confidence
    n_seeds, n_methods, n_folders = len(seeds), len(methods), len(folders)

    data = np.empty((n_folders, len(METHODS), 3))
    data_cis = np.empty((n_folders, len(METHODS), 3))
    data[:] = np.nan
    data_cis[:] = np.nan

    for i, folder in enumerate(folders):

        sequence = main_sequence
        cl_data, ci_data, transfer_data = get_cl_data(methods, metric, seeds, main_sequence, task_length, confidence,
                                                      folder=folder)
        cl_data_alt, ci_data_alt, transfer_data_alt = get_cl_data(methods, metric, seeds, alt_sequence, task_length,
                                                                  confidence, folder=folder)
        baseline_data = get_baseline_data(sequence, seeds, task_length, metric)
        performance = calculate_performance(cl_data)
        # If performance contains nan values, replace those nan values with alternative data to calculate performance
        if np.any(np.isnan(performance)):
            performance_alt = calculate_performance(cl_data_alt)
            for j in range(len(methods)):
                if np.isnan(performance[j]):
                    performance[j] = performance_alt[j]

        performance_ci = calculate_performance(ci_data)
        forgetting = calculate_forgetting(cl_data)
        forgetting_ci = calculate_forgetting(ci_data)
        transfer, transfer_ci = calculate_transfer(transfer_data, baseline_data, len(seeds), confidence)

        for j in range(len(methods)):
            data[i, j] = [performance[j], forgetting[j], transfer[j]]
            data_cis[i, j] = [performance_ci[j], forgetting_ci[j], transfer_ci[j]]

    plot_histograms(data[:, :, 0], folders, methods)

    print('Printing ablation study table\n')
    print_table(folders, methods, data, data_cis, value_cell)
    print('\nPrinting ablations result difference table\n')
    print_table(folders, methods, data, data_cis, diff_cell)


def plot_histograms(data: ndarray, folders: List[str], methods: List[str]):
    plt.style.use('seaborn-deep')
    figsize = (10, 2)
    fig, axes = plt.subplots(1, len(folders) - 1, sharey='all', figsize=figsize)
    width = 0.1  # Adjust the width of bars as needed
    default = data[0]  # Get the default values
    data = data[1:]  # Remove the default values
    folders = folders[1:]  # Remove the default folder

    for i, folder in enumerate(folders):
        for j, method in enumerate(methods):
            diff = ((data[i, j] - default[j]) / default[j]) * 100
            axes[i].bar(i + j * width, [diff], width, label=TRANSLATIONS[method], color=METHOD_COLORS[method])

        axes[i].axhline(0, color='black', lw=1)
        axes[i].set_title(f'{TRANSLATIONS[folder]}')
        axes[i].set_xticks([])

    # Function to format y-labels with a percentage sign
    def format_percent(x, pos):
        return f"{x:.0f}%"

    # Apply the formatting function to the y-labels
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(format_percent))
    axes[0].set_ylabel('Performance Difference')
    plt.ylim(-100, 100)

    fig.legend(*axes[-1].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(methods),
               fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    file_name = 'plots/variations'
    plt.savefig(f'{file_name}.png')
    plt.savefig(f'{file_name}.pdf', dpi=300)
    plt.show()


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
    parser.add_argument("--alt_sequence", type=str, required=False, help="Name of the alternative sequence")
    main(parser.parse_args())
