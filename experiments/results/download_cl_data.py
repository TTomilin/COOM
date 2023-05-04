from typing import List

import argparse
import json
import os
import wandb
from wandb.apis.public import Run

SEQUENCES = {
    'CD4': {
        0: 'default',
        1: 'red',
        2: 'blue',
        3: 'shadows',
    },
    'CD8': {
        0: 'obstacles',
        1: 'green',
        2: 'resized',
        3: 'invulnerable',
        4: 'default',
        5: 'red',
        6: 'blue',
        7: 'shadows',
    },
    'CO4': {
        0: 'chainsaw',
        1: 'raise_the_roof',
        2: 'run_and_gun',
        3: 'health_gathering',
    },
    'CO8': {
        0: 'pitfall',
        1: 'arms_dealer',
        2: 'hide_and_seek',
        3: 'floor_is_lava',
        4: 'chainsaw',
        5: 'raise_the_roof',
        6: 'run_and_gun',
        7: 'health_gathering',
    },
    'COC': {
        0: 'pitfall',
        1: 'arms_dealer',
        2: 'hide_and_seek',
        3: 'floor_is_lava',
        4: 'chainsaw',
        5: 'raise_the_roof',
        6: 'run_and_gun',
        7: 'health_gathering',
    }
}

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

SEPARATE_STORAGE_TAGS = ['REG_CRITIC', 'NO_REG_CRITIC', 'SINGLE_HEAD']


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


def store_data(run: Run, sequence: str, required_metric: str, data_type: str, tags: List[str]) -> None:
    config = json.loads(run.json_config)
    seq_len = 4 if sequence in ['CD4', 'CO4'] else 8
    for env_idx in range(seq_len):
        task = SEQUENCES[sequence][env_idx]
        metric = METRICS[task] if required_metric is None else required_metric
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
        tag = f"{config['wandb_tags']['value'][0].lower()}/" if any(tag in tags for tag in SEPARATE_STORAGE_TAGS) else ''
        path = f'{tag}{sequence}/{method}/seed_{seed}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created new directory {path}")

        file_name = f'{task}_{metric}.json' if data_type == 'test' else f'train_{metric}.json'
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
    parser.add_argument("--type", type=str, default='test', choices=['train', 'test'], help="Type of data to download")
    parser.add_argument("--methods", type=str, default='packnet',
                        choices=['packnet', 'mas', 'vcl', 'agem', 'l2', 'fine-tuning'])
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to store")
    parser.add_argument("--folder", type=str, default=None, help="")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--sequence", type=str, choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'], help="Sequence acronym")
    parser.add_argument("--wandb_tags", type=str, nargs='+', help="WandB tags to filter runs")
    parser.add_argument("--include_runs", type=str, nargs="+", default=[],
                        help="List of runs that shouldn't be filtered out")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
