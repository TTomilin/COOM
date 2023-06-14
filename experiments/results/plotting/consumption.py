import matplotlib as mpl

from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    plt.style.use('seaborn-pastel')
    seeds, sequence, metric = cfg.seeds, cfg.sequence, cfg.metric
    envs = SEQUENCES[sequence]
    methods = METHODS if len(envs) == 4 else METHODS[:-1]
    n_envs, n_seeds, n_methods = len(envs), len(seeds), len(methods)
    fig, ax = plt.subplots(1, 1, sharey='all', sharex='all', figsize=(8, 5))
    n_datapoints = 1

    divider = 1000 if metric == 'memory' else 3600
    data = [get_data_from_file(cfg.metric, n_datapoints, method, seeds, sequence) for method in methods]
    means = [np.nanmean(d) / divider for d in data]
    stds = [np.nanstd(d) / divider for d in data]
    confidence_intervals = CRITICAL_VALUES[cfg.confidence] * np.array(stds) / np.sqrt(
        n_seeds) if metric == 'walltime' else None

    colors = mpl.colormaps['tab20c'].colors[:4] + mpl.colormaps['tab20c'].colors[8:12]
    ax.bar(np.arange(n_methods), means, yerr=confidence_intervals, capsize=4, color=colors)
    ax.set_xticks(np.arange(n_methods))
    ax.set_xticklabels([TRANSLATIONS[method] for method in methods], fontsize=11)
    ax.set_ylabel(TRANSLATIONS[metric], fontsize=13)

    plt.tight_layout()
    file_path = 'plots/system'
    os.makedirs(file_path, exist_ok=True)
    plt.savefig(f'{file_path}/{metric}_{sequence}.png')
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--legend_anchor", type=float, default=0, help="How much to lower the legend")
    main(parser.parse_args())
