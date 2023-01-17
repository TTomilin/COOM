import argparse
import json
import numpy as np
import wandb
from wandb.apis.public import Run

CD4 = {
    0: 'default',
    1: 'red',
    2: 'blue',
    3: 'shadows',
}

CD8 = {
    0: 'obstacles',
    1: 'green',
    2: 'resized',
    3: 'invulnerable',
    4: 'default',
    5: 'red',
    6: 'blue',
    7: 'shadows',
}

CO4 = {
    0: 'chainsaw',
    1: 'raise_the_roof',
    2: 'run_and_gun',
    3: 'health_gathering',
}

CO8 = {
    0: 'pitfall',
    1: 'arms_dealer',
    2: 'hide_and_seek',
    3: 'floor_is_lava',
    4: 'chainsaw',
    5: 'raise_the_roof',
    6: 'run_and_gun',
    7: 'health_gathering',
}


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if run.state == "finished":
            for i in range(4):
                if 'CD4' == args.sequence and 'CD4' in run.url:
                    env = CD4[i]
                    metric_name = f'test/stochastic/{i}/run_and_gun-{env}/{args.metric}'
                    store_data(run, env, metric_name, 'CD4')
                if 'CD8' == args.sequence and 'CD8' in run.url:
                    env = CD8[i]
                    metric_name = f'test/stochastic/{i}/run_and_gun-{env}/{args.metric}'
                    store_data(run, env, metric_name, 'CD8')
                elif 'CO4' == args.sequence and 'CO4' in run.url:
                    env = CO4[i]
                    metric_name = f'test/stochastic/{i}/{env}-default/success'
                    store_data(run, env, metric_name, 'CO4')
                elif 'CO8' == args.sequence and 'CO8' in run.url:
                    env = CO8[i]
                    metric_name = f'test/stochastic/{i}/{env}-default/success'
                    store_data(run, env, metric_name, 'CO8')


def store_data(run: Run, env: str, metric_name: str, sequence: str) -> None:
    history = list(iter(run.scan_history(keys=[metric_name])))
    success = [item[metric_name] for item in history]
    steps = np.arange(1, len(success) + 1) * 1000  # TODO remove 1000
    data = np.transpose([np.zeros(len(success)), steps, success])
    method = get_cl_method(run)
    with open(f'./results/{sequence}/{method}/seed_{run.config["seed"]}/{env}.json', 'w') as f:
        json.dump(data.tolist(), f)


def get_cl_method(run):
    method = run.config["cl_method"]
    if not method:
        method = 'perfect_memory' if run.config['buffer_type'] == 'reservoir' else 'fine_tuning'
    return method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, required=True, choices=['CD4', 'CO4', 'CD8', 'CO8'],
                        help="Name of the task sequence")
    parser.add_argument("--metric", type=str, default='success', help="Name of the metric to store")
    parser.add_argument("--project", type=str, required=True, help="Name of the WandB project")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
