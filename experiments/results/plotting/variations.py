from experiments.results.common import *


def main(cfg: argparse.Namespace) -> None:
    methods, seeds, folders, main_sequence, alt_sequence, metric, task_length, confidence = \
        cfg.methods, cfg.seeds, cfg.folders, cfg.sequence, cfg.alt_sequence, cfg.metric, cfg.task_length, cfg.confidence
    n_seeds, n_methods, n_folders = len(seeds), len(methods), len(folders)

    data = np.empty((len(METHODS), n_folders - 1))
    data[:] = np.nan
    default = None

    for i, folder in enumerate(folders):
        cl_data, ci_data, transfer_data = get_cl_data(methods, metric, seeds, main_sequence, task_length, confidence,
                                                      folder=folder)
        cl_data_alt, ci_data_alt, transfer_data_alt = get_cl_data(methods, metric, seeds, alt_sequence, task_length,
                                                                  confidence, folder=folder)
        performance = calculate_performance(cl_data)
        # If performance contains nan values, replace those nan values with alternative data to calculate performance
        if np.any(np.isnan(performance)):
            performance_alt = calculate_performance(cl_data_alt)
            for j in range(len(methods)):
                if np.isnan(performance[j]):
                    performance[j] = performance_alt[j]

        if folder == 'default':
            default = performance
        else:
            data[i - 1] = performance

    plot_histograms(data, default, folders, methods)


def plot_histograms(data: ndarray, default: ndarray, variations: List[str], methods: List[str]):
    plt.style.use('seaborn-deep')
    figsize = (10, 2)
    fig, axes = plt.subplots(1, len(variations) - 1, sharey='all', figsize=figsize)
    variations = variations[1:]  # Remove the original data folder

    for i, variation in enumerate(variations):
        for j, method in enumerate(methods):
            diff = ((data[i, j] - default[j]) / default[j]) * 100
            axes[i].bar(i + j, [diff], label=TRANSLATIONS[method], color=METHOD_COLORS[method])

        axes[i].axhline(0, color='black', lw=1)
        variation = TRANSLATIONS[variation]
        if variation == 'Critic Regularization':
            variation = 'Critic Reg'
        axes[i].set_title(f'{variation}')
        axes[i].set_xticks([])

    def format_percent(x, pos):
        return f"{x:.0f}%"  # Format the y-labels with a percentage sign

    # Apply the formatting function to the y-labels
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(format_percent))
    axes[0].set_ylabel('Performance Increase')
    plt.ylim(-100, 100)

    fig.legend(*axes[-1].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(methods),
               fancybox=True, shadow=True)
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    file_name = 'plots/variations'
    plt.savefig(f'{file_name}.png')
    plt.savefig(f'{file_name}.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    parser = common_plot_args()
    parser.add_argument("--folders", type=str, required=True, nargs='+', help="Names of the folders")
    parser.add_argument("--alt_sequence", type=str, required=False, help="Name of the alternative sequence")
    main(parser.parse_args())
