import pandas as pd

from experiments.results.common import *


def calculate_data_at_the_end(data):
    return data[:, :, :, -10:].mean(axis=3)


def calculate_forgetting(data: np.ndarray):
    data_at_the_end = calculate_data_at_the_end(data)
    forgetting = (np.diagonal(data_at_the_end, axis1=1, axis2=2) - data_at_the_end[:, :, -1]).clip(0, np.inf)[:,
                 :-1].mean(axis=1)
    return data_at_the_end, forgetting


def calculate_performance(data: np.ndarray):
    data = data.mean(axis=3)
    data = np.triu(data)
    data[data == 0] = np.nan
    return np.nanmean(data, axis=(-1, -2))


def calc_metrics(metric: str, seeds: List[int], sequence: str, task_length: int, confidence: float, second_half: bool):
    envs = SEQUENCES[sequence]
    if second_half:
        envs = envs[len(envs) // 2:]
    n_envs = len(envs)
    iterations = n_envs * task_length
    methods = METHODS if n_envs == 4 or second_half else METHODS[:-1]  # Omit Perfect Memory for 8 env sequences
    cl_data = np.empty((len(methods), n_envs, n_envs, task_length))
    ci_data = np.empty((len(methods), n_envs, n_envs, task_length))
    cl_data[:] = np.nan
    ci_data[:] = np.nan
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
                steps = len(data)
                data = np.array(data).astype(np.float)
                data = np.pad(data, (0, iterations - steps), 'constant', constant_values=np.nan)
                data_per_task = np.array_split(data, n_envs)
                seed_data[k] = data_per_task

            mean = np.nanmean(seed_data, axis=0)
            std = np.nanstd(seed_data, axis=0)
            ci = CRITICAL_VALUES[confidence] * std / np.sqrt(len(seeds))
            cl_data[i][j] = mean
            ci_data[i][j] = ci
    performance = calculate_performance(cl_data)
    performance_ci = calculate_performance(ci_data)
    data_at_the_end, forgetting = calculate_forgetting(cl_data)
    _, forgetting_ci = calculate_forgetting(ci_data)

    # Print performance results
    # print(f'\n\n{sequence}')
    # print_results(performance, performance_ci, methods, "Performance")
    # print_results(forgetting, forgetting_ci, methods, "Forgetting")

    return performance, performance_ci, forgetting, forgetting_ci


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
    performances = np.empty((len(sequences), len(METHODS)))
    performance_cis = np.empty((len(sequences), len(METHODS)))
    forgettings = np.empty((len(sequences), len(METHODS)))
    forgetting_cis = np.empty((len(sequences), len(METHODS)))
    for i, sequence in enumerate(sequences):
        performance, performance_ci, forgetting, forgetting_ci = calc_metrics(cfg.metric, cfg.seeds, sequence,
                                                                              cfg.task_length, cfg.confidence,
                                                                              cfg.second_half)
        performances[i] = np.pad(performance, (0, len(METHODS) - len(performance)), 'constant', constant_values=np.nan)
        performance_cis[i] = np.pad(performance_ci, (0, len(METHODS) - len(performance_ci)), 'constant',
                                    constant_values=np.nan)
        forgettings[i] = np.pad(forgetting, (0, len(METHODS) - len(forgetting)), 'constant', constant_values=np.nan)
        forgetting_cis[i] = np.pad(forgetting_ci, (0, len(METHODS) - len(forgetting_ci)), 'constant',
                                   constant_values=np.nan)
    performance = np.nanmean(performances, axis=0)
    performance_ci = np.nanmean(performance_cis, axis=0)
    forgetting = np.nanmean(forgettings, axis=0)
    forgetting_ci = np.nanmean(forgetting_cis, axis=0)

    print_latex(sequences, performances, performance_cis)
    print_latex(sequences, forgettings, forgetting_cis)
    # print(f'\n\nAverage')
    # print_results(performance, performance_ci, METHODS, "Performance")
    # print_results(forgetting, forgetting_ci, METHODS, "Forgetting")



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
        return ['\\textbf{' + str(val) + '}' if is_max[idx] and not val == '-' else str(val) for idx, val in enumerate(s_arr)]

    results = results.apply(highlight_max, axis=0)
    print(results.to_latex(escape=False))


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--second_half", default=False, action='store_true', help="Only regard sequence 2nd half")
    main(parser.parse_args())
