import argparse
import json
import os
import wandb
from wandb.apis.public import Run

from experiments.results.common import FORBIDDEN_TAGS, common_dl_args, get_cl_method


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args.sequence, args.metric)


def suitable_run(run, args: argparse.Namespace) -> bool:
    # Check whether the run shouldn't be filtered out
    if any(logs in run.name for logs in args.include_runs):
        return True
    # Check whether the run has successfully finished
    if run.state != "finished":
        return False
    # Load the configuration of the run
    config = json.loads(run.json_config)
    # Check whether the provided CL sequence corresponds to the run
    if args.sequence not in run.url:
        return False
    # Check whether the wandb tags are suitable
    if 'wandb_tags' in config:
        tags = config['wandb_tags']['value']
        if any(tag in tags for tag in FORBIDDEN_TAGS):
            return False
    # Check whether the run corresponds to one of the provided seeds
    if args.seeds:
        if 'seed' not in config:
            return False
        seed = config['seed']['value']
        if seed not in args.seeds:
            return False
    if args.method:
        method = get_cl_method(run)
        if method != args.method:
            return False
    # All filters have been passed
    return True


def store_data(run: Run, sequence: str, metric: str) -> None:
    log_key = f'train/{metric}'
    history = list(iter(run.scan_history(keys=[log_key])))

    values = [item[log_key] for item in history]
    method = get_cl_method(run)
    seed = max(run.config["seed"], 1)
    path = f'data/{sequence}/{method}/seed_{seed}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}")

    file_name = f'train_{metric}.json'
    print(f'Saving {run.id} --- {path}/{file_name}')
    with open(f'{path}/{file_name}', 'w') as f:
        json.dump(values, f)


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
