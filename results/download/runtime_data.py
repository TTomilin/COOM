import wandb
from wandb.apis.public import Run

from results.common import *


def main(args: argparse.Namespace) -> None:
    api = wandb.Api()
    filters = build_filters(args)
    runs = api.runs(args.project, filters=filters)
    for run in runs:
        store_data(run, args)


def store_data(run: Run, args: argparse.Namespace) -> None:
    sequence, metric, tags = args.sequence, args.metric, args.wandb_tags
    config = run.config
    if metric == 'walltime':
        system_metric = run.summary['_runtime']
    elif metric == 'memory':
        system_metric = list(iter(run.scan_history(keys=[metric])))[-1][metric]
    else:
        raise ValueError(f'Unsupported system metric: {metric}')
    method = get_cl_method(run)
    seed = config["seed"]
    wandb_tags = config['wandb_tags']
    results_dir = Path(__file__).parent.parent.resolve()
    tag = f"{wandb_tags[0].lower()}" if tags and any(tag in tags for tag in SEPARATE_STORAGE_TAGS) else ''
    path = results_dir / args.data_folder / tag / sequence / method / f'seed_{seed}'
    os.makedirs(path, exist_ok=True)

    file_name = metric if metric == 'walltime' else TRANSLATIONS[metric]
    file_path = f'{path}/{file_name}.json'
    if args.overwrite or not os.path.exists(file_path):
        print(f'Saving {run.id} --- {file_path}')
        with open(f'{file_path}', 'w') as f:
            json.dump([system_metric], f)


if __name__ == "__main__":
    parser = common_dl_args()
    parser.set_defaults(metric="walltime")
    main(parser.parse_args())
