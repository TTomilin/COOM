import os
import wandb
from wandb.apis.public import Run

from experiments.results.common import *


def has_single_tag(run: Run) -> bool:
    config = json.loads(run.json_config)
    if 'wandb_tags' in config:
        tags = config['wandb_tags']['value']
        return 'SINGLE' in tags
    return False


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if has_single_tag(run):
            store_data(run, args)


def store_data(run: Run, args: argparse.Namespace) -> None:
    sequence, metric, seed = args.sequence, args.metric, args.seed
    envs = SEQUENCES[sequence]

    # Load the environment name from the run configuration
    config = json.loads(run.json_config)
    env = config['envs']['value'][0]

    if sequence in ['CD4', 'CD8']:
        scenario = 'run_and_gun'
        task = env
        if scenario not in run.url:
            return
    else:
        scenario = [env for env in envs if env in run.name][0]
        task = scenario
        if env != 'default':
            return

    metric = METRICS[scenario] if metric is None else metric
    path = f'data/single/sac/seed_{seed}'
    file_path = f'{path}/{task}_{metric}.json'
    if not args.overwrite and os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping")
        return
    log_key = f'train/{metric}'
    history = list(iter(run.scan_history(keys=[log_key])))
    values = [item[log_key] for item in history][:args.task_length]
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}")
    print(f'Saving {file_path}')
    with open(file_path, 'w') as f:
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
    parser = common_dl_args()
    parser.add_argument("--seed", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Seed of the run")
    main(parser.parse_args())
