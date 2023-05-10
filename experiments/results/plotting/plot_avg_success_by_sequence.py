import os

from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    plt.rcParams['axes.grid'] = True
    seeds, metric, sequences = cfg.seeds, cfg.metric, cfg.sequences
    n_sequences, n_seeds = len(sequences), len(seeds)
    fig, ax = plt.subplots(n_sequences, 1, sharey='all', sharex='all', figsize=(10, 6))
    max_steps = -np.inf
    colors = COLORS[cfg.sequence]

    for l, sequence in enumerate(sequences):
        envs = SEQUENCES[sequence]
        n_envs = len(envs)
        methods = METHODS if n_envs == 4 else METHODS[:-1]
        iterations = cfg.task_length * n_envs
        for i, method in enumerate(methods):
            seed_data = np.empty((n_envs, n_seeds, iterations))
            seed_data[:] = np.nan
            for j, env in enumerate(envs):
                for k, seed in enumerate(seeds):
                    path = os.path.join(os.getcwd(), '../data', sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                    if not os.path.exists(path):
                        continue
                    with open(path, 'r') as f:
                        data = json.load(f)
                    steps = len(data)
                    max_steps = max(max_steps, steps)
                    seed_data[j, k, np.arange(steps)] = data

            plot_curve(ax[l], cfg.confidence, colors[i], TRANSLATIONS[method], iterations, seed_data, n_seeds * n_envs,
                       agg_axes=(0, 1))

        ax[l].set_ylabel('Average Success')
        ax[l].set_title(sequence, fontsize=14)
        add_task_labels(ax[l], envs, iterations, n_envs)

    folder = 'success'
    os.makedirs(folder, exist_ok=True)
    plot_name = f'{folder}/{"_".join(sequences)}'
    plot_and_save(ax[-1], plot_name, n_col=len(methods[0]), legend_anchor=-0.5)


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
