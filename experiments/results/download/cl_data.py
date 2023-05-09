import json
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
            store_data(run, args.sequence, args.metric, args.type, args.wandb_tags)


def suitable_run(run, args: argparse.Namespace) -> bool:
    # Check whether the run shouldn't be filtered out
    if any(logs in run.name for logs in args.include_runs):
        return True
    # Check whether the run has successfully finished
    if run.state != "finished":
        return False
    # Load the configuration of the run
    config = json.loads(run.json_config)
    # Check whether the provided CL sequence corresponds with the run
    if args.sequence not in run.url:
        return False
    # Check whether the run includes one of the provided wandb tags
    if args.wandb_tags:
        # Tag(s) are provided but not listed in the run
        if 'wandb_tags' not in config:
            return False
        tags = config['wandb_tags']['value']
        # Check whether the run includes one of the provided tags in args.tags
        if not any(tag in tags for tag in args.wandb_tags):
            return False
    # All filters have been passed
    return True


def store_data(run: Run, sequence: str, metric: str, data_type: str, tags: List[str]) -> None:
    config = json.loads(run.json_config)
    seq_len = 4 if sequence in ['CD4', 'CO4'] else 8
    for env_idx in range(seq_len):
        task = SEQUENCES[sequence][env_idx]
        metric = METRICS[task] if metric == 'env' else metric
        env = f'run_and_gun-{task}' if sequence in ['CD4', 'CD8'] else f'{task}-{ENVS[sequence]}'
        log_key = f'test/stochastic/{env_idx}/{env}/{metric}' if data_type == 'test' else f'train/{metric}'
        history = list(iter(run.scan_history(keys=[log_key])))

        # Legacy
        if not history:
            print(f'No data for {run.name} {env}')
            env = f'seek_and_slay-default' if sequence in ['CO4'] else f'seek_and_slay-{task}'
            log_key = f'test/stochastic/{env_idx}/seek_and_slay-{task}/{metric}'
            history = list(iter(run.scan_history(keys=[log_key])))

            # More legacy
            if not history:
                print(f'Still no data for {run.name} {env}')
                log_key = f'test/stochastic/{env_idx}/seek_and_slay-shadows_obstacles/{metric}'
                history = list(iter(run.scan_history(keys=[log_key])))

        values = [item[log_key] for item in history]
        method = get_cl_method(run)
        seed = max(run.config["seed"], 1)
        tag = f"{config['wandb_tags']['value'][0].lower()}/" if any(
            tag in tags for tag in SEPARATE_STORAGE_TAGS) else ''
        path = f'{tag}{sequence}/{method}/seed_{seed}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created new directory {path}")

        file_name = f'{task}_{metric}.json' if data_type == 'test' else f'train_{metric}.json'
        print(f'Saving {run.id} --- {file_name}')
        with open(f'{path}/{file_name}', 'w') as f:
            json.dump(values, f)


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
