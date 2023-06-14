import pandas as pd
from numpy import ndarray
from typing import Tuple, Callable

from experiments.results.common import *


def calculate_performance(data: np.ndarray):
    data = data.mean(axis=3)
    data = np.triu(data)
    data[data == 0] = np.nan
    return np.nanmean(data, axis=(-1, -2))


def calculate_transfer(transfer_data, baseline_data, n_seeds: int, confidence: float) -> Tuple[ndarray, ndarray]:
    auc_cl = np.nanmean(transfer_data, axis=-1)
    auc_baseline = np.nanmean(baseline_data, axis=-1)
    ft = (auc_cl - auc_baseline) / (1 - auc_baseline)
    ft_mean = np.nanmean(ft, 0)
    ft_std = np.nanstd(ft, 0)
    ci = CRITICAL_VALUES[confidence] * ft_std / np.sqrt(n_seeds)
    return ft_mean, ci


def get_cl_data(folder: str, metric: str, seeds: List[int], sequence: str, task_length: int, confidence: float):
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    iterations = n_envs * task_length
    methods = METHODS if sequence in ['CD4', 'CO4'] else METHODS[:-1]
    cl_data = np.empty((len(methods), n_envs, n_envs, task_length))
    ci_data = np.empty((len(methods), n_envs, n_envs, task_length))
    transfer_data = np.empty((len(seeds), len(methods), task_length * n_envs))
    cl_data[:] = np.nan
    ci_data[:] = np.nan
    transfer_data[:] = np.nan
    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            seed_data = np.empty((len(seeds), n_envs, task_length))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), 'data', folder, sequence, method, f'seed_{seed}',
                                    f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                task_start = j * task_length
                steps = len(data)
                data = np.array(data).astype(np.float)
                data = np.pad(data, (0, iterations - steps), 'constant', constant_values=np.nan)
                data_per_task = np.array_split(data, n_envs)
                seed_data[k] = data_per_task
                transfer_data[k, i, np.arange(task_start, task_start + task_length)] = data[
                                                                                       task_start: task_start + task_length]
            mean = np.nanmean(seed_data, axis=0)
            std = np.nanstd(seed_data, axis=0)
            ci = CRITICAL_VALUES[confidence] * std / np.sqrt(len(seeds))
            cl_data[i][j] = mean
            ci_data[i][j] = ci
    return cl_data, ci_data, transfer_data


def main(cfg: argparse.Namespace) -> None:
    methods, seeds, folders, sequence, metric = cfg.methods, cfg.seeds, cfg.folders, cfg.sequence, cfg.metric

    methods = METHODS if sequence in ['CD4', 'CO4'] else METHODS[:-1]
    envs = SEQUENCES[sequence]
    n_envs, n_seeds, n_methods, n_folders = len(envs), len(seeds), len(methods), len(folders)

    data = np.empty((n_folders, len(METHODS), 2))
    data_cis = np.empty((n_folders, len(METHODS), 2))
    data[:] = np.nan
    data_cis[:] = np.nan

    for i, folder in enumerate(folders):
        cl_data, ci_data, transfer_data = get_cl_data(folder, cfg.metric, cfg.seeds, sequence, cfg.task_length,
                                                      cfg.confidence)
        baseline_data = get_baseline_data(sequence, seeds, cfg.task_length, cfg.metric)
        performance = calculate_performance(cl_data)
        performance_ci = calculate_performance(ci_data)
        transfer, transfer_ci = calculate_transfer(transfer_data, baseline_data, len(seeds), cfg.confidence)

        for j in range(len(methods)):
            data[i, j] = [performance[j], transfer[j]]
            data_cis[i, j] = [performance_ci[j], transfer_ci[j]]

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
        default_ap = data[0, i, 0]  # Default Average Performance
        default_ft = data[0, i, 1]  # Default Forward Transfer
        for j, sequence in enumerate(folders):
            cell_values = []
            for k in range(data_types):
                value = data[j, i, k]
                ci = data_cis[j, i, k]
                cell_string = get_cell_func(value, ci, j, k, default_ap, default_ft)
                cell_values.append(cell_string)
            cell = ' & '.join(cell_values)
            row.append(cell)
        results.loc[len(results)] = row
    results = results.set_index('Method')
    multi_col_format = 'c@{\hskip 0.05in}c@{\hskip 0.05in}c'
    latex_table = results.to_latex(escape=False, column_format='l' + multi_col_format * (len(folders)))
    print(latex_table)


def value_cell(value: float, ci: float, *args) -> str:
    return f'{value:.2f} \tiny ± {ci:.2f}' if not np.isnan(value) else '-'


def diff_cell(value: float, _: float, j: int, k: int, default_ap: ndarray, default_ft: ndarray) -> str:
    diff_string = ((value - default_ap) / default_ap) * 100 if k == 0 else value - default_ft
    return f'{value:.2f}' if j == 0 else f' {diff_string:+.2f}' + (r'\%' if k == 0 else '')


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--folders", type=str, required=True, nargs='+', help="Names of the folders")
    main(parser.parse_args())
