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
    sequence, metric, data_type, tags = args.sequence, args.metric, args.type, args.wandb_tags
    config = json.loads(run.json_config)
    seq_len = 1 if data_type == 'train' else 4 if sequence in ['CD4', 'CO4'] else 8 if sequence in ['CD8', 'CO8'] else 16
    for env_idx in range(seq_len):
        task = SEQUENCES[sequence][env_idx]
        if metric == 'env':
            metric = METRICS[task]
        env = f'run_and_gun-{task}' if sequence in ['CD4', 'CD8', 'CD16'] else f'{task}-{ENVS[sequence]}'
        log_key = f'test/stochastic/{env_idx}/{env}/{metric}' if data_type == 'test' else f'train/{metric}'
        history = list(iter(run.scan_history(keys=[log_key])))

        # Legacy
        if not history:
            # print(f'No data for {run.name} {env}')
            env = f'seek_and_slay-default' if sequence in ['CO4'] else f'seek_and_slay-{task}'
            log_key = f'test/stochastic/{env_idx}/{env}/{metric}'
            history = list(iter(run.scan_history(keys=[log_key])))

            # More legacy
            if not history:
                # print(f'Still no data for {run.name} {env}')
                log_key = f'test/stochastic/{env_idx}/seek_and_slay-shadows_obstacles/{metric}'
                history = list(iter(run.scan_history(keys=[log_key])))

        values = [item[log_key] for item in history]
        method = get_cl_method(run)
        seed = max(run.config["seed"], 1)
        wandb_tags = config['wandb_tags']['value']
        tag = f'{next((tag for tag in tags if tag in wandb_tags), None).lower()}/' if tags and any(tag in tags for tag in SEPARATE_STORAGE_TAGS) else ''
        path = f'{tag}{sequence}/{method}/seed_{seed}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created new directory {path}")

        file_name = f'{task}_{metric}.json' if data_type == 'test' else f'train_{metric}.json'
        if seq_len == 16:
            file_name = f'{env_idx // 8}_{file_name}'
        file_path = f'{path}/{file_name}'
        if args.overwrite or not os.path.exists(file_path):
            print(f'Saving {run.id} --- {path}/{file_name}')
            with open(f'{path}/{file_name}', 'w') as f:
                json.dump(values, f)


if __name__ == "__main__":
    parser = common_dl_args()
    main(parser.parse_args())
