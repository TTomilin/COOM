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
    'seek_and_slay': 'kills',  # Legacy name
    'health_gathering': 'ep_length',
}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if run.id in args.run_ids:
            store_data(run, args.metric)


def store_data(run: Run, required_metric: str) -> None:
    scenarios = METRICS.keys()
    task = [scenario for scenario in scenarios if scenario in run.name][0]
    metric = METRICS[task] if required_metric is None else required_metric
    log_key = f'train/{metric}'
    history = list(iter(run.scan_history(keys=[log_key])))
    values = [item[log_key] for item in history][:200]
    file_name = f'./single/{task}_{metric}.json'
    with open(file_name, 'w') as f:
        json.dump(values, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to store")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--run_ids", type=str, nargs="+", default=[], help="List of experiment names to downloaded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
