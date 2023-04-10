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


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    sequence = args.sequence
    for run in runs:
        config = json.loads(run.json_config)
        if run.state == "finished" and sequence in run.url and valid_COC(args, config) or any(logs in run.name for logs in args.failed_runs):
            store_data(run, sequence, args.metric, args.type)


def valid_COC(args: argparse.Namespace, config: dict) -> bool:
    if args.sequence != 'COC':
        return True
    tags = config['wandb_tags']['value']
    return tags and 'SIMPLIFIED_ENVS' in tags[0]


def store_data(run: Run, sequence: str, required_metric: str, data_type: str) -> None:
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
        path = f'./{sequence}/{method}/seed_{seed}'
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
    parser.add_argument("--methods", type=str, default='packnet', choices=['packnet', 'mas', 'vcl', 'agem', 'l2', 'fine-tuning'])
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to store")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--sequence", type=str, choices=['CD4', 'CO4', 'CD8', 'CO8', 'COC'], help="Sequence acronym")
    parser.add_argument("--failed_runs", type=str, nargs="+", default=[],
                        help="List of experiment names that don't have a 'finished' status, but ought to be downloaded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
