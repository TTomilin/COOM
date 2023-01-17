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
    'perfect_memory': 'Perfect Memory'
}


def main(args: argparse.Namespace) -> None:
    methods = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']
    fig, ax = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(8, 3))
    max_steps = -np.inf

    for i, method in enumerate(methods):
        path = os.path.join(args.cl_logs, f'{method}.json')
        with open(path, 'r') as f:
            data = json.load(f)
            data = np.array(data)[:, 1:]  # Remove timestamp
            max_steps = max(max_steps, len(data))

            x = data[:, 0] // 1000
            y = data[:, 1]
            y = gaussian_filter1d(y, sigma=2)
            ax.plot(x, y, label=translations[method])

    ax.set_xlabel("Timesteps (K)")
    ax.set_ylabel("Success Rate")
    ax.legend(loc='lower right')
    ax.set_title("Training")

    n_envs = len(args.envs)
    env_steps = max_steps // n_envs
    env_name_locations = np.arange(0 + env_steps // 2, max_steps + env_steps // 2, env_steps)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(env_name_locations)
    ax2.set_xticklabels(args.envs)
    ax2.set_xlabel(r"Environment")
    ax2.tick_params(axis='both', which='both', length=0)

    fig.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cl_logs", type=str, default='results/cross_scenario/train')
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
