import argparse
import json
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
    }
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
        if run.state == "finished" and sequence in run.url or any(logs in run.name for logs in args.failed_runs):
            store_data(run, sequence, args.metric)


def store_data(run: Run, sequence: str, required_metric: str) -> None:
    seq_len = 4 if sequence in ['CD4', 'CO4'] else 8
    for env_idx in range(seq_len):
        task = SEQUENCES[sequence][env_idx]
        metric = METRICS[task] if required_metric is None else required_metric
        env = f'{task}-default' if sequence in ['CO4', 'CO8'] else f'run_and_gun-{task}'
        log_key = f'test/stochastic/{env_idx}/{env}/{metric}'
        history = list(iter(run.scan_history(keys=[log_key])))
        if not history:
            print(f'No data for {run.name} {env}')
            log_key = f'test/stochastic/{env_idx}/seek_and_slay-default/{metric}'
            history = list(iter(run.scan_history(keys=[log_key])))
        values = [item[log_key] for item in history]
        method = get_cl_method(run)
        seed = max(run.config["seed"], 1)
        file_name = f'./results/{sequence}/{method}/seed_{seed}/{task}_{metric}.json'
        with open(file_name, 'w') as f:
            json.dump(values, f)


def get_cl_method(run):
    method = run.config["cl_method"]
    if not method:
        method = 'perfect_memory' if run.config['buffer_type'] == 'reservoir' else 'fine_tuning'
    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default=None, help="Name of the metric to store")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    parser.add_argument("--sequence", type=str, required=True, choices=['CD4', 'CO4', 'CD8', 'CO8'],
                        help="Name of the task sequence")
    parser.add_argument("--failed_runs", type=str, nargs="+", default=[],
                        help="List of experiment names that don't have a 'finished' status, but ought to be downloaded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
