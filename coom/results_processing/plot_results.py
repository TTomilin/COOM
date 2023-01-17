import argparse
import json
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

translations = {
    'packnet': 'PackNet',
    'mas': 'MAS',
    'agem': 'AGEM',
    'l2': 'L2',
    'vcl': 'VCL',
    'fine_tuning': 'Fine-tuning',
    'perfect_memory': 'Perfect Memory',

    'pitfall': 'Pitfall',
    'arms_dealer': 'Arms Dealer',
    'hide_and_seek': 'Hide and Seek',
    'floor_is_lava': 'Floor is Lava',
    'chainsaw': 'Chainsaw',
    'raise_the_roof': 'Raise the Roof',
    'run_and_gun': 'Run and Gun',
    'health_gathering': 'Health Gathering',

    'obstacles': 'Obstacles',
    'green': 'Green',
    'resized': 'Resized',
    'invulnerable': 'Invulnerable',
    'default': 'Default',
    'red': 'Red',
    'blue': 'Blue',
    'shadows': 'Shadows',
}

seq_envs = {
    'CD4': ['default', 'red', 'blue', 'shadows'],
    'CD8': ['obstacles', 'green', 'resized', 'invulnerable', 'default', 'red', 'blue', 'shadows'],
    'CO4': ['chainsaw', 'raise_the_roof', 'run_and_gun', 'health_gathering'],
    'CO8': ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun', 'health_gathering'],
}


def main(args: argparse.Namespace) -> None:
    plt.style.use('seaborn')
    methods = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']
    seeds = ['1', '2', '3']
    cl_data = {}
    fig, ax = plt.subplots(len(methods), 1, sharey=True, sharex=True, figsize=(10, 15))
    envs = seq_envs[args.sequence]
    env_names = [translations[e] for e in envs]
    max_steps = -np.inf
    iterations = 800 if args.sequence in ['CD4', 'CO4'] else 1600

    for i, method in enumerate(methods):
        for env in envs:
            seed_data = np.empty((len(seeds), iterations))
            seed_data[:] = np.nan
            for j, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), args.sequence, method, f'seed_{seed}', f'{env}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                data = np.array(data)[:, 1:]  # Remove timestamp
                max_steps = max(max_steps, len(data))
                cl_data[f'{method}_{env}'] = data
                steps = ((data[:, 0] // 1000) - 1).astype(int)
                # print(f'{method}_{env}_{seed}: {len(steps)}')
                seed_data[j, steps] = data[:, 1]

            y = np.nanmean(seed_data, axis=0)
            # print(f'{method}_{env} nan count: {np.isnan(y).sum()}')
            y = gaussian_filter1d(y, sigma=2)
            # Plot confidence intervals
            if args.use_ci:
                ci = np.nanstd(seed_data, axis=0)
                ci = gaussian_filter1d(ci, sigma=2)
                ax[i].fill_between(np.arange(iterations), y - ci, y + ci, alpha=0.2)  # TODO CIs go below 0 and above 1
            ax[i].plot(y, label=env)
            ax[i].tick_params(labelbottom=True)

        ax[i].set_ylabel("Success Rate")
        ax[i].set_title(translations[method])
        handles, labels = ax[i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

    n_envs = len(envs)
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
    parser.add_argument("--sequence", type=str, required=True, choices=['CD4', 'CO4', 'CD8', 'CO8'],
                        help="Name of the task sequence")
    parser.add_argument(
        "--use_ci",
        type=bool,
        default=True,
        help="Show confidence intervals for every plot (may be significantly slower to generate)"
    )
    parser.add_argument("--output_path", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
