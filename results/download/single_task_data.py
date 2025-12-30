import wandb
from wandb.apis.public import Run

from results.common import *


def has_single_tag(run: Run) -> bool:
    """To collect baseline data, we run the jobs with the 'SINGLE' wandb tag."""
    config = run.config
    if 'wandb_tags' in config:
        tags = config['wandb_tags']
        return 'SINGLE' in tags
    return False


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters)
    for run in runs:
        if has_single_tag(run):
            store_data(run, args)


def store_data(run: Run, args: argparse.Namespace) -> None:
    sequence, metric = args.sequence, args.metric
    envs = SEQUENCES[sequence]

    # Load the environment name from the run configuration
    config = run.config
    env = config['envs'][0]

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
    seed = config['seed']
    results_dir = Path(__file__).parent.parent.resolve()
    path = results_dir / args.data_folder / 'single' / 'sac' / f'seed_{seed}'
    os.makedirs(path, exist_ok=True)
    file_path = path / f'{task}_{metric}.json'
    if not args.overwrite and os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping")
        return
    log_key = f'train/{metric}'
    history = list(iter(run.scan_history(keys=[log_key])))
    values = [item[log_key] for item in history][:args.task_length]
    print(f'Saving {file_path}')
    with open(file_path, 'w') as f:
        json.dump(values, f)


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
