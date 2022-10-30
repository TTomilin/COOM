import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from coom.utils.utils import str2bool

translations = {
    'packnet': 'PackNet',
    'mas': 'MAS',
    'agem': 'AGEM',
    'l2': 'L2',
    'vcl': 'VCL',
    'fine_tuning': 'Fine-tuning',
    'perfect_memory': 'Perfect Memory',

    'chainsaw': 'Chainsaw',
    'raise_the_roof': 'Raise the Roof',
    'seek_and_slay': 'Seek and Slay',
    'health_gathering': 'Health Gathering',

    'default': 'Default',
    'red': 'Red',
    'blue': 'Blue',
    'shadows_obstacles': 'Obstacles',
}


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    methods = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']
    cl_data = {}
    fig, ax = plt.subplots(len(methods), 1, sharey=True, sharex=True, figsize=(10, 15))
    env_names = [translations[e] for e in args.envs]
    max_steps = -np.inf

    for i, method in enumerate(methods):
        for env in args.envs:
            path = os.path.join(args.cl_logs, method, f'{env}.json')
            with open(path, 'r') as f:
                data = json.load(f)
            data = np.array(data)[:, 1:]  # Remove timestamp
            max_steps = max(max_steps, len(data))
            cl_data[f'{method}_{env}'] = data

            x = data[:, 0] // 1000
            y = data[:, 1]
            y = gaussian_filter1d(y, sigma=2)
            ax[i].plot(x, y, label=env)
            ax[i].tick_params(labelbottom=True)

        ax[i].set_ylabel("Success Rate")
        ax[i].set_title(translations[method])
        handles, labels = ax[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

    n_envs = len(args.envs)
    env_steps = max_steps // n_envs
    env_name_locations = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

    ax2 = ax[0].twiny()
    ax2.set_xlim(ax[0].get_xlim())
    ax2.set_xticks(env_name_locations)
    ax2.set_xticklabels(env_names)
    ax2.tick_params(axis='both', which='both', length=0)

    ax[-1].set_xlabel("Timesteps (K)")
    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cl_logs", type=str, default='results/cross_scenario')
    parser.add_argument("--envs", type=str, nargs="+", required=True, help="Name of the environments to plot")
    parser.add_argument(
        "--use_ci",
        type=str2bool,
        default=False,
        help="When True, confidence intervals are shown for every plot. Note that plots may be significantly "
             "slower to generate."
    )
    parser.add_argument("--output_path", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
