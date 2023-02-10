import argparse

import logging
import time
import wandb
from typing import List


def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param exceptions: Lists of exceptions that trigger a retry attempt
    :type exceptions: Tuple of Exceptions
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    print(f'Exception thrown when attempting to run {func}, attempt {attempt} out of {times}')
                    time.sleep(min(2**attempt, 10))
                    attempt += 1

            return func(*args, **kwargs)

        return newfn

    return decorator


class WandBLogger:

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser):
        def arg(*args, **kwargs):
            parser.add_argument(*args, **kwargs)

        arg('--wandb_entity', default=None, type=str, help='WandB username (entity).')
        arg('--wandb_project', default='COOM', type=str, help='WandB "Project"')
        arg('--wandb_group', default=None, type=str, help='WandB "Group". Name of the env by default.')
        arg('--wandb_job_type', default='train', type=str, help='WandB job type')
        arg('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
        arg('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
        arg('--wandb_dir', default=None, type=str, help='the place to save WandB files')
        arg('--wandb_experiment', default='', type=str, help='Identifier to specify the experiment')

    def __init__(self, parser: argparse.ArgumentParser, scenarios: List[str], timestamp: str, sequence: str = ''):
        """ Call WandB initialization before summary writer, otherwise sync_tensorboard does not work. """
        args = parser.parse_args()
        self.with_wandb = args.with_wandb

        if not args.with_wandb:
            logging.info('Weights and Biases integration disabled')
            return

        if args.wandb_group is None:
            args.wandb_group = scenarios[0] if len(scenarios) == 1 else 'Cross-Scenario'

        if args.cl_method:
            method = args.cl_method
        elif len(scenarios) == 1:
            method = 'sac'
        elif args.buffer_type == 'reservoir':
            method = 'perfect_memory'
        else:
            method = 'fine_tuning'
        args.wandb_unique_id = f'{method}_seed_{args.seed}_{args.wandb_group}_{sequence}_{args.wandb_experiment}_{timestamp}'

        logging.info(
            f'Weights and Biases integration enabled. Project: {args.wandb_project}, user: {args.wandb_entity}, '
            f'group: {args.wandb_group}, unique_id: {args.wandb_unique_id}')

        # Try multiple times, as this occasionally fails
        @retry(3, exceptions=(Exception,))
        def init_wandb_func():
            wandb.init(
                dir=args.wandb_dir,
                project=args.wandb_project,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                id=args.wandb_unique_id,
                name=args.wandb_unique_id,
                group=args.wandb_group,
                job_type=args.wandb_job_type,
                tags=args.wandb_tags,
                resume=False,
                settings=wandb.Settings(start_method='fork'),
            )

        logging.info('Initializing WandB...')
        try:
            if args.wandb_key:
                wandb.login(key=args.wandb_key)
            init_wandb_func()
        except Exception as exc:
            logging.error(f'Could not initialize WandB! {exc}')

        wandb.config.update(args, allow_val_change=True)

    def finish_wandb(self):
        if self.with_wandb:
            import wandb
            wandb.run.finish()
