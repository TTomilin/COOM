import argparse
import json
import os
import wandb
from wandb.apis.public import Run

ENVS = {
    'CO4': 'default',
    'CO8': 'default',
    'COC': 'hard',
}

METRICS = {
    'pitfall': 'distance',
    'arms_dealer': 'arms_dealt',
    'hide_and_seek': 'ep_length',
    'floor_is_lava': 'ep_length',
    'chainsaw': 'kills',
    'raise_the_roof': 'ep_length',
    'run_and_gun': 'kills',
    'health_gathering': 'ep_length',
}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    sequence = args.sequence
    for run in runs:
        config = json.loads(run.json_config)
        if run.state == "finished" and sequence in run.url and valid_COC(args, config) or any(logs in run.name for logs in args.failed_runs):
            store_data(run, sequence, args.metric)


def valid_COC(args: argparse.Namespace, config: dict) -> bool:
    if args.sequence != 'COC':
        return True
    tags = config['wandb_tags']['value']
    sequence = config['sequence']['value']
    method = config['cl_method']['value']
    return tags and 'SIMPLIFIED_ENVS' in tags[0] or sequence == 'COC' and method == 'vcl'


def store_data(run: Run, sequence: str, metric: str) -> None:
    metric = metric
    log_key = f'train/{metric}'
    history = list(iter(run.scan_history(keys=[log_key])))

    values = [item[log_key] for item in history]
    method = get_cl_method(run)
    seed = max(run.config["seed"], 1)
    path = f'./{sequence}/{method}/seed_{seed}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}")

    file_name = f'train_{metric}.json'
    print(f'Saving {run.id} --- {file_name}')
    with open(f'{path}/{file_name}', 'w') as f:
        json.dump(values, f)


def get_cl_method(run):
    method = run.config["cl_method"]
    if not method:
        method = 'perfect_memory' if run.config['buffer_type'] == 'reservoir' else 'fine_tuning'
    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to store")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--sequence", type=str, choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'], help="Sequence acronym")
    parser.add_argument("--failed_runs", type=str, nargs="+", default=[],
                        help="List of experiment names that don't have a 'finished' status, but ought to be downloaded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
