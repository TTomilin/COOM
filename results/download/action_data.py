import wandb
from wandb.apis.public import Run

from results.common import *


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters)
    base_dir = Path(__file__).parent.parent.resolve()
    for run in runs:
        store_data(base_dir, run, args.sequence, args.test_envs, args.eval_mode, args.n_actions, args.data_folder)


def store_data_for_env(base_dir: Path, run: Run, sequence: str, eval_mode: str, n_actions: int, data_folder: str,
                       test_env: int = None) -> None:
    if test_env is not None:
        task = SEQUENCES[sequence][test_env]
        env = f'run_and_gun-{task}' if sequence in ['CD4', 'CD8'] else f'{task}-{ENVS[sequence]}'
        log_key = f'test/{eval_mode}/{test_env}/{env}/actions'
    else:
        log_key = 'train/actions'
    log_keys = [f'{log_key}/{i}' for i in range(n_actions)]
    history = list(iter(run.scan_history(keys=log_keys)))
    if not history:
        return
    actions = np.array([[log[f'{log_key}/{i}'] for i in range(n_actions)] for log in history])

    method = get_cl_method(run)
    seed = run.config["seed"]
    folder = 'train' if test_env is None else f'test_{test_env}'
    path = base_dir / data_folder / 'actions' / sequence / method / folder
    os.makedirs(path, exist_ok=True)

    file_path = path / f'seed_{seed}.json'
    print(f'Saving {run.id} run actions to {file_path}')
    with open(file_path, 'w') as f:
        json.dump(actions.tolist(), f)


def store_data(base_dir: Path, run: Run, sequence: str, test_envs: List[int], eval_mode: str, n_actions: int,
               data_folder: str) -> None:
    for env in (test_envs or [None]):
        store_data_for_env(base_dir, run, sequence, eval_mode, n_actions, data_folder, env)


def action_dl_args() -> argparse.ArgumentParser:
    parser = common_dl_args()
    parser.add_argument("--n_actions", type=int, default=12,
                        help="Number of discrete actions that the models were trained with")
    return parser


if __name__ == "__main__":
    parser = action_dl_args()
    main(parser.parse_args())
