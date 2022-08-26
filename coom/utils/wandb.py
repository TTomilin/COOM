import logging
import time
from datetime import datetime

import wandb


def retry(times, exceptions):
    """
    Retry Decorator https://stackoverflow.com/a/64030200/1645784
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
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


def init_wandb(cfg):
    """
    Must call initialization of Wandb before summary writer is initialized, otherwise
    sync_tensorboard does not work.
    """

    if not cfg.with_wandb:
        logging.info('Weights and Biases integration disabled')
        return

    if cfg.scenario:
        cfg.wandb_group = cfg.scenario

    if 'wandb_unique_id' not in cfg:
        # if we're going to restart the experiment, this will be saved to a json file
        cfg.wandb_unique_id = f'{cfg.algorithm}_seed_{cfg.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

    logging.info(
        f'Weights and Biases integration enabled. Project: {cfg.wandb_project}, user: {cfg.wandb_user}, group: {cfg.wandb_group}, unique_id: {cfg.wandb_unique_id}')

    # this can fail occasionally, so we try a couple more times
    @retry(3, exceptions=(Exception,))
    def init_wandb_func():
        wandb.init(
            dir=cfg.wandb_dir,
            project=cfg.wandb_project, entity=cfg.wandb_user, sync_tensorboard=True,
            id=cfg.wandb_unique_id,
            name=cfg.wandb_unique_id,
            group=cfg.wandb_group, job_type=cfg.wandb_job_type, tags=cfg.wandb_tags,
            # resume=True,
            resume=False,
            settings=wandb.Settings(start_method='fork'),
        )

    logging.info('Initializing WandB...')
    try:
        if cfg.wandb_key:
            wandb.login(key=cfg.wandb_key)
        init_wandb_func()
    except Exception as exc:
        logging.error(f'Could not initialize WandB! {exc}')

    wandb.config.update(cfg, allow_val_change=True)


def finish_wandb(cfg):
    if cfg.with_wandb:
        import wandb
        wandb.run.finish()
