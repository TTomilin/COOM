import logging
import time
import wandb
from argparse import Namespace

from coom.ope.lib.constants import DOOM_GAMES


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


def init_wandb(job, args: Namespace):
    """
    Must call initialization of WandB before summary writer is initialized, otherwise sync_tensorboard does not work.
    """

    if not args.with_wandb:
        logging.info('Weights and Biases integration disabled')
        return

    if args.wandb_group is None:
        args.wandb_group = args.scenarios[0] if len(args.scenarios) == 1 else 'Cross-Scenario'

    if 'wandb_unique_id' not in args:
        if 'scenarios' in args.game:
            method = args.cl_method if args.cl_method else 'sac'
            args.wandb_unique_id = f'{method}_seed_{args.seed}_{args.wandb_group}_{args.wandb_experiment}_{args.timestamp}'
        else:
            args.wandb_unique_id = f'{args.game}_{args.experiment_name}_{args.domain}_{job}_{args.wandb_experiment}_{args.timestamp}'

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


def finish_wandb(cfg):
    if cfg.with_wandb:
        import wandb
        wandb.run.finish()
