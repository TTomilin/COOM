from typing import Dict

import os

import json

import argparse
import numpy as np


TRANSLATIONS = {
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

    'success': 'Success Rate',
    'kills': 'Kill Count',
}

SEQUENCES = {
    'CD4': ['default', 'red', 'blue', 'shadows'],
    'CD8': ['obstacles', 'green', 'resized', 'invulnerable', 'default', 'red', 'blue', 'shadows'],
    'CO4': ['chainsaw', 'raise_the_roof', 'run_and_gun', 'health_gathering'],
    'CO8': ['pitfall', 'arms_dealer', 'hide_and_seek', 'floor_is_lava', 'chainsaw', 'raise_the_roof', 'run_and_gun',
            'health_gathering'],
}

METHODS = ['packnet', 'mas', 'agem', 'l2', 'vcl', 'fine_tuning', 'perfect_memory']


def calculate_forward_transfer(data, baseline_data, normalize=True):
    data = data.copy()

    task_num_to_name = get_task_num_to_name(data)
    steps_per_task = int(data.steps_per_task.unique())

    long_baseline = []
    for env in sorted(data["train/active_env"].unique()):
        #         if np.isnan(env): continue
        env = int(env)
        env_name = task_num_to_name[env]

        # baseline
        current_baseline = baseline_data[baseline_data["task"] == env_name].copy()
        current_baseline["current_success"] = current_baseline[
            f"test/stochastic/0/{env_name}/success"
        ]
        current_baseline["x"] += env * steps_per_task
        current_baseline["train/active_env"] = env
        long_baseline += [current_baseline]

        # current task: update data with 'current_succes' column
        env_indices = data["train/active_env"] == env
        current_col = data.columns[
            data.columns.str.contains(f"test/stochastic/{env}/.*/success", regex=True)
        ][0]
        data.loc[env_indices, "current_success"] = data.loc[env_indices, current_col]

    long_baseline = pd.concat(long_baseline)

    # correct for double seeds
    #     unique_exps = data.groupby(['experiment_id', 'seed'], as_index=False).size()
    #     target_size = 500 if cw10 else 1000
    #     unique_exps = unique_exps[unique_exps['size'] == target_size].drop_duplicates(subset='seed', keep="last")['experiment_id']
    #     data = data[data['experiment_id'].isin(unique_exps)].reset_index()
    # display(data.groupby(['experiment_id', 'seed'], as_index=False).size())

    data = (
        data.drop("x", axis=1)
        .groupby(["train/active_env", "experiment_id"])["current_success"]
        .mean()
        .reset_index()
    )
    long_baseline = (
        long_baseline.drop("x", axis=1)
        .groupby(["train/active_env", "experiment_id"])["current_success"]
        .mean()
        .reset_index()
    )

    # ugly
    X = data.pivot(
        index="experiment_id", columns="train/active_env", values="current_success"
    ).to_numpy()
    Y = long_baseline.pivot(
        index="experiment_id", columns="train/active_env", values="current_success"
    ).reset_index(drop=True)
    Y = pd.DataFrame({c: Y[c].dropna().values for c in Y.columns}).to_numpy()

    T = X.shape[1]

    ranges = {f"[{i}]": range(i, i + 1) for i in range(T)}
    ranges.update(
        {
            f"[{0}:{T // 2}]": range(0, T // 2),
            f"[{T // 2}:{T}]": range(T // 2, T),
            f"[{0}:{T}]": range(0, T),
        }
    )

    BCI = BootstrapCI(
        X=X, Y=Y, num_bootstrap=4000, confidence=0.9, statistics=statistics, ranges=ranges, seed=0
    )
    CIs = BCI.ci()

    ci_result = defaultdict(list)

    for env in sorted(data["train/active_env"].unique()):
        ci_result["train/active_env"].append(env)

        for name in statistics.keys():
            lb, ub = CIs[name][f"[{int(env)}]"]
            m = BCI.original_data_metrics[name][f"[{int(env)}]"]

            ci_result[f"lower_bound_{name}"].append(lb)
            ci_result[f"upper_bound_{name}"].append(ub)
            ci_result[f"CI_{name}"].append(f"{m:.2f} [{lb:.2f}, {ub:.2f}]")

            lbfh, ubfh = CIs[name][f"[{0}:{T // 2}]"]
            lbsh, ubsh = CIs[name][f"[{T // 2}:{T}]"]
            lbt, ubt = CIs[name][f"[{0}:{T}]"]
            mfh = BCI.original_data_metrics[name][f"[{0}:{T // 2}]"]
            msh = BCI.original_data_metrics[name][f"[{T // 2}:{T}]"]
            mt = BCI.original_data_metrics[name][f"[{0}:{T}]"]

            ci_result[f"lb_first_half_{name}"].append(lbfh)
            ci_result[f"ub_first_half_{name}"].append(ubfh)
            ci_result[f"CI_first_half_{name}"].append(f"{mfh:.2f} [{lbfh:.2f}, {ubfh:.2f}]")

            ci_result[f"lb_second_half_{name}"].append(lbsh)
            ci_result[f"ub_second_half_{name}"].append(ubsh)
            ci_result[f"CI_second_half_{name}"].append(f"{msh:.2f} [{lbsh:.2f}, {ubsh:.2f}]")

            ci_result[f"lb_total_{name}"].append(lbt)
            ci_result[f"ub_total_{name}"].append(ubt)
            ci_result[f"CI_total_{name}"].append(f"{mt:.2f} [{lbt:.2f}, {ubt:.2f}]")
            ci_result[f"total_{name}"].append(mt)

    ci_result = pd.DataFrame(ci_result)

    # We have all the data inside - best place for confidence interval analysis :)
    data = data.merge(long_baseline, on="train/active_env", suffixes=("", "_baseline"))
    data = data.groupby("train/active_env").mean()
    data = data.merge(ci_result, on="train/active_env")

    data["ft"] = data["current_success"] - data["current_success_baseline"]
    data["normalized_ft"] = data["ft"] / (1 - data["current_success_baseline"])
    #     for name in statistics.keys():
    #         data[f'{name}_CI'] = data[f'{name}'].map('{:.2f}'.format) + " " + data[f'CI_{name}']

    return data


def calculate_data_at_the_end(data):
    return data[:, :, :, -10:].mean(axis=3)


def calculate_forgetting(data: np.ndarray):
    data_at_the_end = calculate_data_at_the_end(data)
    forgetting = (np.diagonal(data_at_the_end, axis1=1, axis2=2) - data_at_the_end[:, :, -1]).clip(0, np.inf)[:, :-1].mean(axis=1)
    return data_at_the_end, forgetting


def main(args: argparse.Namespace) -> None:
    seeds = ['1', '2', '3']
    sequence = args.sequence
    metric = args.metric
    task_length = args.task_length
    envs = SEQUENCES[sequence]
    n_envs = len(envs)
    iterations = n_envs * task_length
    methods = METHODS if n_envs == 4 else METHODS[:-1]  # Omit Perfect Memory for 8 env sequences
    cl_data = np.empty((len(methods), n_envs, n_envs, task_length))
    ci_data = np.empty((len(methods), n_envs, n_envs, task_length))
    cl_data[:] = np.nan
    ci_data[:] = np.nan
    for i, method in enumerate(methods):
        for j, env in enumerate(envs):
            seed_data = np.empty((len(seeds), n_envs, task_length))
            seed_data[:] = np.nan
            for k, seed in enumerate(seeds):
                path = os.path.join(os.getcwd(), sequence, method, f'seed_{seed}', f'{env}_{metric}.json')
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    data = json.load(f)
                steps = len(data)
                data = np.pad(data, (0, iterations - steps), 'constant', constant_values=np.nan)
                data_per_task = np.array_split(data, n_envs)
                seed_data[k] = data_per_task
                # print(f'{method}_{env}_{seed}: {len(steps)}')

            y = np.nanmean(seed_data, axis=0)
            ci = np.nanstd(seed_data, axis=0)
            cl_data[i][j] = y
            ci_data[i][j] = ci
    data_at_the_end, forgetting = calculate_forgetting(cl_data)
    _, ci_forget = calculate_forgetting(ci_data)
    # Join mean and std and normalize results
    joined_results = np.array((forgetting, ci_forget))
    joined_results = joined_results / np.linalg.norm(joined_results)
    # Print results
    for i, method in enumerate(methods):
        forget = joined_results[0][i]
        ci = joined_results[1][i]
        print(f'{method}: {forget :.2f} ({ci :.2f})')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, choices=['CD4', 'CO4', 'CD8', 'CO8'],
                        help="Name of the task sequence")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to calculate forgetting")
    parser.add_argument("--task_length", type=int, default=200, help="Number of iterations x 1000 per task")
    parser.add_argument("--output_path", type=str, default="results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
