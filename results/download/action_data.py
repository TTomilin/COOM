import wandb
from wandb.apis.public import Run

from results.common import *


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args.sequence, args.test_envs)


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


def store_data_for_env(run: Run, sequence: str, test_env: int) -> None:
    n_actions = 12
    if test_env is not None:
        task = SEQUENCES[sequence][test_env]
        env = f'run_and_gun-{task}' if sequence in ['CD4', 'CD8'] else f'{task}-{ENVS[sequence]}'
        log_key = f'test/stochastic/{test_env}/{env}/actions'
    else:
        log_key = 'train/actions'
    log_keys = [f'{log_key}/{i}' for i in range(n_actions)]
    history = list(iter(run.scan_history(keys=log_keys)))
    if not history:
        return
    actions = np.array([[log[f'{log_key}/{i}'] for i in range(n_actions)] for log in history])

    method = get_cl_method(run)
    seed = json.loads(run.json_config)["seed"]["value"]
    folder = 'train' if test_env is None else f'test_{test_env}'
    path = f'actions/{sequence}/{method}/{folder}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}")

    file_name = f'{path}/seed_{seed}.json'
    print(f'Saving {run.id} run actions to {file_name}')
    with open(file_name, 'w') as f:
        json.dump(actions.tolist(), f)


def store_data(run: Run, sequence: str, test_envs: List[int]) -> None:
    if test_envs:
        for env in test_envs:
            store_data_for_env(run, sequence, env)
    else:
        store_data_for_env(run, sequence, None)


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
