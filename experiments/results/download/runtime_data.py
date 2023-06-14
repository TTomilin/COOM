import wandb
from wandb.apis.public import Run

from experiments.results.common import *


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    # Project is specified by <entity/project-name>
    runs = api.runs(args.project)
    for run in runs:
        if suitable_run(run, args):
            store_data(run, args)


def store_data(run: Run, args: argparse.Namespace) -> None:
    sequence, metric, tags = args.sequence, args.metric, args.wandb_tags
    config = json.loads(run.json_config)
    history = list(iter(run.scan_history(keys=[metric])))
    if not history:
        print(f'No data named "{metric}" found for {run.name}')
        return
    values = [item[metric] for item in history]
    if metric == 'buffer_capacity':
        values *= 350
    method = get_cl_method(run)
    seed = max(run.config["seed"], 1)
    wandb_tags = config['wandb_tags']['value']
    tag = f"{wandb_tags[0].lower()}/" if tags and any(tag in tags for tag in SEPARATE_STORAGE_TAGS) else ''
    path = f'{tag}{sequence}/{method}/seed_{seed}'
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created new directory {path}")

    file_path = f'{path}/{metric}.json'
    if args.overwrite or not os.path.exists(file_path):
        print(f'Saving {run.id} --- {file_path}')
        with open(f'{file_path}', 'w') as f:
            json.dump(values, f)


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
