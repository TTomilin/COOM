import pandas as pd

from experiments.results.common import *


def get_cl_data(seeds, envs: List[str], task_length, metric: str, sequence: str):
    cl_data = np.empty((len(seeds), len(METHODS), task_length * len(envs)))
    cl_data[:] = np.nan
    for k, seed in enumerate(seeds):
        for i, method in enumerate(METHODS):
            for j, env in enumerate(envs):
                path = f'{os.getcwd()}/data/{sequence}/{method}/seed_{seed}/{env}_{metric}.json'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = j * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    cl_data[k, i, np.arange(task_start, task_start + steps)] = data
    return cl_data


def print_latex(sequences, mean, ci):
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


def main(cfg: argparse.Namespace) -> None:
    seeds, sequences = cfg.seeds, cfg.sequences
    means, cis = [], []
    for sequence in sequences:
        envs = SEQUENCES[sequence]
        task_length = cfg.task_length

        cl_data = get_cl_data(seeds, envs, task_length, cfg.metric, sequence)
        baseline_data = get_baseline_data(sequence, envs, seeds, task_length, cfg.metric)

        auc_cl = np.nanmean(cl_data, axis=-1)
        auc_baseline = np.nanmean(baseline_data, axis=-1)

        ft = (auc_cl - auc_baseline) / (1 - auc_baseline)
        ft_mean = np.nanmean(ft, 0)
        ft_std = np.nanstd(ft, 0)
        ci = CRITICAL_VALUES[cfg.confidence] * ft_std / np.sqrt(len(seeds))
        means.append(ft_mean)
        cis.append(ci)
    print_latex_swapped(sequences, np.array(means), np.array(cis), highlight_max=True)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
