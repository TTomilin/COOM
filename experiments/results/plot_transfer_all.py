import json
import os
from matplotlib import pyplot as plt

from experiments.results.common import *

COLOR_SAC = '#C44E52'
COLORS = ['#1F77B4', '#55A868', '#4C72B0', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']


def get_baseline_data(sequence: str, scenarios: List[str], seeds: List[str], task_length: int,
                      set_metric: str = None) -> np.ndarray:
    seed_data = np.empty((len(seeds), task_length * len(scenarios)))
    seed_data[:] = np.nan
    baseline_type = 'single_hard' if sequence == 'COC' else 'single'
    for i, env in enumerate(scenarios):
        metric = set_metric if set_metric else METRICS[env]
        for k, seed in enumerate(seeds):
            path = f'{os.getcwd()}/data/{baseline_type}/sac/seed_{seed}/{env}_{metric}.json'
            if not os.path.exists(path):
                continue
            with open(path, 'r') as f:
                data = json.load(f)[0: task_length]
            steps = len(data)
            start = i * task_length
            seed_data[k, np.arange(start, start + steps)] = data
    baseline_data = np.nanmean(seed_data, axis=0)
    baseline_data = gaussian_filter1d(baseline_data, sigma=2)
    return baseline_data


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds = cfg.seeds
    sequence = cfg.sequence
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    methods = METHODS if n_envs == 4 else METHODS[:-1]
    n_methods = len(methods)
    fig, ax = plt.subplots(n_methods, 1, sharex='all', sharey='all', figsize=(9, 12))
    task_length = cfg.task_length
    iterations = task_length * n_envs
    baseline = get_baseline_data(sequence, envs, seeds, task_length, cfg.metric)

    for i, method in enumerate(methods):
        seed_data = np.empty((len(seeds), iterations))
        seed_data[:] = np.nan
        for j, env in enumerate(envs):
            metric = cfg.metric if cfg.metric else METRICS[env]
            for k, seed in enumerate(seeds):
                path = f'{os.getcwd()}/data/{sequence}/{method}/seed_{seed}/{env}_{metric}.json'
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    task_start = j * task_length
                    data = json.load(f)[task_start: task_start + task_length]
                    steps = len(data)
                    seed_data[k, np.arange(task_start, task_start + steps)] = data

        mean = np.nanmean(seed_data, axis=0)
        mean = gaussian_filter1d(mean, sigma=2)

        ax[i].plot(mean, label=TRANSLATIONS[method], color=COLORS[i])
        ax[i].plot(baseline, label=TRANSLATIONS['sac'], color=COLOR_SAC)
        ax[i].tick_params(labelbottom=True)
        ax[i].fill_between(np.arange(iterations), mean, baseline, where=(mean < baseline), alpha=0.3, color=COLOR_SAC,
                           interpolate=True)
        ax[i].fill_between(np.arange(iterations), mean, baseline, where=(mean >= baseline), alpha=0.3, color=COLORS[i],
                           interpolate=True)

        ax[i].set_ylabel(TRANSLATIONS[metric], fontsize=11)
        ax[i].set_title(TRANSLATIONS[method], fontsize=11)
        ax[i].set_ylim([0, 1])

    add_task_labels(ax, envs, iterations, n_envs)

    handles, labels = [], []
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    legend_anchor = -1.1 if n_envs == 4 else -0.7
    n_cols = n_envs if n_envs == 4 else n_methods + 1
    ax[-1].set_xlabel("Timesteps (K)")
    ax[-1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, legend_anchor), ncol=n_cols, fancybox=True,
                  shadow=True)
    fig.tight_layout()
    plt.savefig(f'plots/{sequence}_transfer.png')
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
