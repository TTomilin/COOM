import os

import argparse
import json
import wandb
from wandb.apis.public import Run

METRICS = {
    'pitfall': 'distance',
    'arms_dealer': 'arms_dealt',
    'hide_and_seek': 'ep_length',
    'floor_is_lava': 'ep_length',
    'chainsaw': 'kills',
    'raise_the_roof': 'ep_length',
    'run_and_gun': 'kills',
    'seek_and_slay': 'kills',  # Legacy scenario name
    'health_gathering': 'ep_length',
}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if run.id in args.run_ids:
            store_data(run, args.metric, args.seed)


def store_data(run: Run, required_metric: str, seed: str) -> None:
    scenarios = METRICS.keys()
    scenario = [scenario for scenario in scenarios if scenario in run.name]
    if not scenario:
        print(f"Could not find task for run {run.name}")
        scenario = 'run_and_gun'
    else:
        scenario = scenario[0]
    metric = METRICS[scenario] if required_metric is None else required_metric
    log_key = f'train/{metric}'
    history = list(iter(run.scan_history(keys=[log_key])))
    values = [item[log_key] for item in history][:200]
    path = f'./single/sac/seed_{seed}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}")
    file_name = f'{path}/{scenario}_{metric}.json'
    print(f'Saving {file_name}')
    with open(file_name, 'w') as f:
        json.dump(values, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to store")
    parser.add_argument("--env", type=str, default='default', help="Name of the Doom environment")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--seed", type=str, required=True, choices=['1', '2', '3'], help="Seed of the run")
    parser.add_argument("--run_ids", type=str, nargs="+", default=[], help="List of experiment names to downloaded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
