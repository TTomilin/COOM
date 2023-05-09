import os

import argparse
import json
import wandb
from wandb.apis.public import Run

from experiments.results.common import METRICS


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        config = json.loads(run.json_config)
        if run.state == "finished" and is_hard_env(args, config):
            store_data(run, args.metric, args.seed)


def is_hard_env(args: argparse.Namespace, config: dict) -> bool:
    tags = config['wandb_tags']['value']
    return 'envs' in config and args.env in config['envs']['value'][0] and tags and 'HARD' in tags[0]


def store_data(run: Run, required_metric: str, seed: str) -> None:
    scenarios = METRICS.keys()
    scenario = [scenario for scenario in scenarios if scenario in run.name]
    if not scenario:
        print(f"Could not find scenario for run {run.name}")
        scenario = 'run_and_gun'
    else:
        scenario = scenario[0]
    metric = METRICS[scenario] if required_metric is None else required_metric
    log_key = f'train/{metric}'
    history = list(iter(run.scan_history(keys=[log_key])))
    values = [item[log_key] for item in history][:200]
    path = f'./single_hard/sac/seed_{seed}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}")
    file_name = f'{scenario}_{metric}.json'
    print(f'Saving {run.id} --- {file_name}')
    with open(f'{path}/{file_name}', 'w') as f:
        json.dump(values, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to store")
    parser.add_argument("--env", type=str, default='default', help="Name of the Doom environment")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--seed", type=str, required=True, choices=['1', '2', '3'], help="Seed of the run")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
