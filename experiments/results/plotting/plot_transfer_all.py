from experiments.results.common import *
from experiments.results.common import get_baseline_data

COLOR_SAC = '#C44E52'
COLORS = ['#1F77B4', '#55A868', '#4C72B0', '#8172B2', '#CCB974', '#64B5CD', '#777777', '#917113']


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-deep')
    seeds, sequence, methods = cfg.seeds, cfg.sequence, cfg.methods
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    if n_envs == 8:
        methods = [method for method in methods if method != 'perfect_memory']
    n_methods = len(methods)
    fig, ax = plt.subplots(n_methods, 1, sharex='all', sharey='all', figsize=(9, 4))
    task_length = cfg.task_length
    iterations = task_length * n_envs
    baseline = get_baseline_data(sequence, envs, seeds, task_length, cfg.metric)
    baseline = gaussian_filter1d(baseline, sigma=2)

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

        ax[i].plot(mean, label=method, color=COLORS[i])
        ax[i].plot(baseline, label='sac', color=COLOR_SAC)
        ax[i].tick_params(labelbottom=True)
        ax[i].fill_between(np.arange(iterations), mean, baseline, where=(mean < baseline), alpha=0.3, color=COLOR_SAC,
                           interpolate=True)
        ax[i].fill_between(np.arange(iterations), mean, baseline, where=(mean >= baseline), alpha=0.3, color=COLORS[i],
                           interpolate=True)

        # ax[i].set_ylabel('Current Task Success', fontsize=11)
        ax[i].set_title(TRANSLATIONS[method], fontsize=13)
        ax[i].set_xlim([0, iterations])
        ax[i].set_ylim([0, 1])
        ax[i].grid(True, which='major', axis='x', linestyle='--')

    add_task_labels(ax[0], envs, iterations, n_envs)
    main_ax = add_main_ax(fig)
    main_ax.set_ylabel('Current Task Success', fontsize=11, labelpad=20)

    handles, labels = [], []
    for a in ax:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    sac_idx = labels.index('sac')
    handles.append(handles.pop(sac_idx))
    labels.append(labels.pop(sac_idx))
    labels = [TRANSLATIONS[label] for label in labels]

    legend_anchor = -1.1 if n_envs == 4 else -0.7
    n_cols = n_envs if n_envs == 4 else n_methods + 1
    # ax[-1].set_xlabel("Timesteps (K)", fontsize=11)
    ax[-1].legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.9), ncol=n_cols, fancybox=True,
                  shadow=True)
    plt.tight_layout(rect=[0, -0.25, 1, 1])
    plt.savefig(f'plots/transfer/{sequence}_packnet_l2.png')
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    main(parser.parse_args())
