from experiments.results.common import *
from experiments.results.common import get_baseline_data

COLOR_SAC = '#C44E52'
COLORS = ['#1F77B4', '#55A868', '#4C72B0', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#FF8C00', '#917113']


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds, sequence, methods, metric = cfg.seeds, cfg.sequence, cfg.methods, cfg.metric
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    if methods is None:
        methods = METHODS if n_envs == 4 else METHODS[:-1]
    n_methods = len(methods)
    figsize = (12, 12) if n_methods > 1 else (10, 2)
    fig, ax = plt.subplots(n_methods, 1, sharex='all', sharey='all', figsize=figsize)
    task_length = cfg.task_length
    n_data_points = task_length * n_envs
    iterations = n_data_points * LOG_INTERVAL
    baseline = get_baseline_data(sequence, seeds, task_length, cfg.metric)
    baseline = gaussian_filter1d(baseline, sigma=KERNEL_SIGMA)

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
        mean = gaussian_filter1d(mean, sigma=KERNEL_SIGMA)
        x = np.arange(0, iterations, LOG_INTERVAL)
        cur_ax.plot(x, mean, label=method, color=COLORS[i])
        cur_ax.plot(x, baseline, label='sac', color=COLOR_SAC)
        cur_ax.fill_between(x, mean, baseline, where=(mean < baseline), alpha=0.3, color=COLOR_SAC, interpolate=True)
        cur_ax.fill_between(x, mean, baseline, where=(mean >= baseline), alpha=0.3, color=COLORS[i], interpolate=True)

        if n_methods > 1:
            cur_ax.set_title(TRANSLATIONS[method], fontsize=13)
        cur_ax.set_xlim([0, iterations])
        cur_ax.set_ylim([0, 1])
        cur_ax.grid(True, which='major', axis='x', linestyle='--')

    top_ax = ax if n_methods == 1 else ax[0]
    bottom_ax = ax if n_methods == 1 else ax[-1]
    bottom_ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 2))
    bottom_ax.tick_params(labelbottom=True)
    add_task_labels(top_ax, envs, iterations, n_envs, fontsize=12)
    main_ax = add_main_ax(fig, fontsize=13)
    main_ax.set_ylabel('Current Task Success', fontsize=13, labelpad=25)

    handles, labels = ax.get_legend_handles_labels() if n_methods == 1 else get_handles_and_labels(ax)
    sac_idx = labels.index('sac')
    handles.append(handles.pop(sac_idx))
    labels.append(labels.pop(sac_idx))
    labels = [TRANSLATIONS[label] for label in labels]

    if n_methods == 1:
        bottom_ax.legend(handles, labels)
    else:
        anchor = -1.2 if n_methods > 1 else -0.55
        n_cols = n_envs if n_envs == 4 else n_methods + 1
        bottom_ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, anchor), ncol=n_cols, fancybox=True,
                         shadow=True)
    bottom_adjust = -0.05 if n_methods > 1 else 0
    plt.tight_layout(rect=[0, bottom_adjust, 1, 1])
    file_path = 'plots/transfer'
    os.makedirs(file_path, exist_ok=True)
    file_name = f'{file_path}/{sequence}.png' if not cfg.methods else f'{file_path}/{sequence}_{"_".join(methods)}.png'
    plt.savefig(file_name)
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
