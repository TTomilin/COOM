from experiments.results.common import *
from experiments.results.common import get_baseline_data

COLOR_SAC = '#C44E52'
COLORS = ['#1F77B4', '#55A868', '#4C72B0', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds, sequence, methods, metric = cfg.seeds, cfg.sequence, cfg.methods, cfg.metric
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    if n_envs == 8:
        methods = [method for method in methods if method != 'perfect_memory']
    n_methods = len(methods)
    figsize = (9, 4) if n_methods > 1 else (10, 2)
    fig, ax = plt.subplots(n_methods, 1, sharex='all', sharey='all', figsize=figsize)
    task_length = cfg.task_length
    iterations = task_length * n_envs * LOG_INTERVAL
    baseline = get_baseline_data(sequence, seeds, task_length, cfg.metric)
    baseline = gaussian_filter1d(baseline, sigma=2)

    for i, method in enumerate(methods):
        cur_ax = ax if n_methods == 1 else ax[i]
        seed_data = np.empty((len(seeds), int(iterations / LOG_INTERVAL)))
        seed_data[:] = np.nan
        for j, env in enumerate(envs):
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
        x = np.arange(0, iterations, LOG_INTERVAL)
        cur_ax.plot(x, mean, label=method, color=COLORS[i])
        cur_ax.plot(x, baseline, label='sac', color=COLOR_SAC)
        cur_ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 2))
        cur_ax.fill_between(x, mean, baseline, where=(mean < baseline), alpha=0.3, color=COLOR_SAC, interpolate=True)
        cur_ax.fill_between(x, mean, baseline, where=(mean >= baseline), alpha=0.3, color=COLORS[i], interpolate=True)

        # ax[i].set_ylabel('Current Task Success', fontsize=11)
        if n_methods > 1:
            cur_ax.set_title(TRANSLATIONS[method], fontsize=13)
        cur_ax.set_xlim([0, iterations])
        cur_ax.set_ylim([0, 1])
        cur_ax.grid(True, which='major', axis='x', linestyle='--')

    top_ax = ax if n_methods == 1 else ax[0]
    bottom_ax = ax if n_methods == 1 else ax[-1]
    add_task_labels(top_ax, envs, iterations, n_envs, fontsize=10)
    main_ax = add_main_ax(fig)
    main_ax.set_ylabel('Current Task Success', fontsize=10, labelpad=25)

    handles, labels = ax.get_legend_handles_labels() if n_methods == 1 else get_handles_and_labels(ax)
    sac_idx = labels.index('sac')
    handles.append(handles.pop(sac_idx))
    labels.append(labels.pop(sac_idx))
    labels = [TRANSLATIONS[label] for label in labels]

    if n_methods == 1:
        bottom_ax.legend(handles, labels)
    else:
        anchor = -0.9 if n_methods > 1 else -0.55
        n_cols = n_envs if n_envs == 4 else n_methods + 1
        bottom_ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, anchor), ncol=n_cols, fancybox=True,
                         shadow=True)
    bottom_adjust = -0.25 if n_methods > 1 else 0
    plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
    plt.savefig(f'plots/transfer/{sequence}_{"_".join(methods)}.pdf')
    plt.show()


def get_handles_and_labels(ax):
    handles, labels = [], []
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    return handles, labels


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
