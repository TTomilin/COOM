import os
import wandb
from wandb.apis.public import Run

from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args.sequence, args.metric)


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
