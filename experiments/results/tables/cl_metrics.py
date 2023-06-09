import pandas as pd
from numpy import ndarray
from typing import Tuple

from experiments.results.common import *


def calculate_data_at_the_end(data):
    return data[:, :, :, -10:].mean(axis=3)


def calculate_forgetting(data: np.ndarray):
    end_data = calculate_data_at_the_end(data)
    forgetting = (np.diagonal(end_data, axis1=1, axis2=2) - end_data[:, :, -1]).clip(0, np.inf)
    return forgetting


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


def get_cl_data(metric: str, seeds: List[int], sequence: str, task_length: int, confidence: float, second_half: bool):
    envs = SEQUENCES[sequence]
    if second_half:
        envs = envs[len(envs) // 2:]
    n_envs = len(envs)
    iterations = n_envs * task_length
    methods = METHODS if n_envs == 4 or second_half else METHODS[:-1]  # Omit Perfect Memory for 8 env sequences
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
                path = os.path.join(os.getcwd(), 'data', sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                if second_half:
                    data = data[len(data) // 2:]
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


def print_results(metric_data: np.ndarray, ci: np.ndarray, methods: List[str], metric: str):
    print(f'\n{"Method":<20}{metric:<20}{"CI":<20}')
    for i, method in enumerate(methods):
        print(f'{method:<20}{metric_data[i]:<20.2f}{ci[i]:.2f}')


def normalize(metric_data, ci):
    joined_results = np.array((metric_data, ci))
    joined_results = joined_results / np.linalg.norm(joined_results)
    return joined_results


def main(cfg: argparse.Namespace) -> None:
    seeds, sequences = cfg.seeds, cfg.sequences

    # Create 3-dimensional arrays to store performances, forgettings, transfers + their confidence intervals
    data = np.empty((len(sequences), len(METHODS), 3))
    data_cis = np.empty((len(sequences), len(METHODS), 3))
    data[:] = np.nan
    data_cis[:] = np.nan

    for i, sequence in enumerate(sequences):
        methods = METHODS if sequence in ['CD4', 'CO4'] else METHODS[:-1]
        cl_data, ci_data, transfer_data = get_cl_data(cfg.metric, cfg.seeds, sequence, cfg.task_length, cfg.confidence,
                                                      cfg.second_half)
        baseline_data = get_baseline_data(sequence, seeds, cfg.task_length, cfg.metric)
        performance = calculate_performance(cl_data)
        performance_ci = calculate_performance(ci_data)
        forgetting = calculate_forgetting(cl_data)
        forgetting_ci = calculate_forgetting(ci_data)
        transfer, transfer_ci = calculate_transfer(transfer_data, baseline_data, len(seeds), cfg.confidence)

        if sequence == cfg.forgetting_sequence:
            print_task_forgetting(methods, cfg.forgetting_sequence, forgetting, forgetting_ci)
        for j in range(len(methods)):
            data[i, j] = [performance[j], forgetting[:, :-1].mean(axis=1)[j], transfer[j]]
            data_cis[i, j] = [performance_ci[j], forgetting_ci[:, :-1].mean(axis=1)[j], transfer_ci[j]]

    print_combined(sequences, data, data_cis)


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


def print_combined(sequences, data, data_cis):
    data_types = data.shape[-1]
    pd.set_option('display.max_colwidth', None)
    results = pd.DataFrame(columns=['Method'] + [f'\multicolumn{{{data_types}}}{{c}}{{{sequence}}}' for sequence in sequences] + ['\multicolumn{3}{c}{Average}'])
    highlight_func = [np.nanmax if k in [0, 2] else np.nanmin for k in range(3)]
    for i, method in enumerate(METHODS):
        row = [TRANSLATIONS[method]]
        for j, sequence in enumerate(sequences):
            cell_values = []
            for k in range(data_types):
                value = data[j, i, k]
                significant = highlight_func[k](data[j, :, k]) == value
                cell_string = f'\\textbf{{{value:.2f}}}' if significant else f'{value:.2f}' if not np.isnan(value) else '-'
                cell_values.append(cell_string)
            cell = ' & '.join(cell_values)
            row.append(cell)
        cell_values = []
        for k in range(data_types):
            value = np.nanmean(data[:, i, k])
            significant = highlight_func[k](np.nanmean(data[:, :, k], axis=0)) == value
            avg_string = f'\\textbf{{{value:.2f}}}' if significant else f'{value:.2f}' if not np.isnan(value) else '-'
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
    parser.add_argument("--forgetting_sequence", type=str, default='CO8', choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'], help="Sequence to compare the forgetting of tasks on")
    main(parser.parse_args())
