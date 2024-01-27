"""

Some simple logging functionality, inspired by rllab's logging.

Logs to a tab-separated-values file (path/to/output_directory/progress.txt)

"""
import argparse
import atexit
import json
import logging
import os
import os.path as osp
import time
from typing import List

import numpy as np
import tensorflow as tf
import wandb

from CL.utils.running import get_readable_timestamp, get_random_string
from CL.utils.serialization import convert_json

color2num = dict(
    gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append("1")
    return "\x1b[%sm%s\x1b[0m" % (";".join(attr), string)


class Logger:
    """
    A general-purpose logger.

    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(
            self,
            logger_output,
            config,
            group_id,
            output_dir=None,
            output_fname="progress.tsv",
            exp_name=None,
            with_mrunner=False,
    ):
        """
        Initialize a Logger.

        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``./experiments/somerandomnumber``.

            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.

            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        self.logger_output = logger_output

        run_id = get_readable_timestamp() + "_" + get_random_string()
        self.output_dir = output_dir or f"./logs/{group_id}/{run_id}"
        if osp.exists(self.output_dir):
            print(f"Warning: Log dir {self.output_dir} already exists! Storing info there anyway.")
        else:
            os.makedirs(self.output_dir)

        self.output_file = None
        if "tsv" in self.logger_output:
            self.output_file = open(osp.join(self.output_dir, output_fname), "w")
            atexit.register(self.output_file.close)

        if "neptune" in self.logger_output:
            if with_mrunner:
                import mrunner

                self._neptune_exp = mrunner.helpers.client_helper.experiment_
            else:
                import neptune

                neptune.init()  # env variable NEPTUNE_PROJECT is used
                self._neptune_exp = neptune.create_experiment()

        if "tensorboard" in self.logger_output:
            self.tb_writer = tf.summary.create_file_writer(self.output_dir)
            self.tb_writer.set_as_default()

        self.save_config(config)

        print(colorize(f"Logging data to {self.output_dir}", "green", bold=True))

        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color="green"):
        """Print a colorized message to stdout."""
        # Add a formatted date time to the message
        msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
        print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        # Allow new statistics to be introduced when switching to a task from a different scenario
        if key not in self.log_headers:
            self.log_headers.append(key)
        if key in self.log_current_row:
            val = np.mean([val, self.log_current_row[key]])
            print(f"Warning: Overwriting {key} = {self.log_current_row[key]:.2f} with new average {val}:.2f")
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.

        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).

        Example use:

        .. code-block:: python

            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        output = json.dumps(config_json, separators=(",", ":\t"), indent=4, sort_keys=True)
        print(colorize("Saving config:\n", color="cyan", bold=True))
        print(output)
        with open(osp.join(self.output_dir, "config.json"), "w") as out:
            out.write(output)

    def setup_tf_saver(self, sess, inputs, outputs):
        """
        Set up easy model saving for tensorflow.

        Call once, after defining your computation graph but before training.

        Args:
            sess: The Tensorflow session in which you train your computation
                graph.

            inputs (dict): A dictionary that maps from keys of your choice
                to the tensorflow placeholders that serve as inputs to the
                computation graph. Make sure that *all* of the placeholders
                needed for your outputs are included!

            outputs (dict): A dictionary that maps from keys of your choice
                to the outputs from your computation graph.
        """
        self.tf_saver_elements = dict(session=sess, inputs=inputs, outputs=outputs)
        self.tf_saver_info = {
            "inputs": {k: v.name for k, v in inputs.items()},
            "outputs": {k: v.name for k, v in outputs.items()},
        }

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15, max(key_lens))
        keystr = "%" + "%d" % max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-" * n_slashes)
        step = self.log_current_row.get("total_env_steps")
        scalar_start = time.time()
        for key in self.log_headers:
            val = self.log_current_row.get(key, 0.0)
            valstr = "%8.3g" % val if hasattr(val, "__float__") else val
            print(fmt % (key, valstr))
            vals.append(val)

            # Log to Neptune
            if "neptune" in self.logger_output:
                # Try several times.
                for _ in range(10):
                    try:
                        self._neptune_exp.send_metric(key, step, val)
                    except:
                        time.sleep(5)
                    else:
                        break
            if "tensorboard" in self.logger_output:
                tf.summary.scalar(key, data=val, step=step)

        print("Scalar logging time: ", time.time() - scalar_start)

        if "tensorboard" in self.logger_output:
            flush_start = time.time()
            tf.summary.flush()
            print(f"Flushed tensorboard in {time.time() - flush_start:.2f} seconds")
        print("-" * n_slashes, flush=True)
        if self.output_file is not None:
            write_out_start = time.time()
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
            print(f"Wrote to output file in {time.time() - write_out_start:.2f} seconds")
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.

    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.

    With an EpochLogger, each time the quantity is calculated, you would
    use

    .. code-block:: python

        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use

    .. code-block:: python

        epoch_logger.log_tabular(NameOfQuantity, **options)

    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, d):
        """
        Save something into the epoch_logger's current state.

        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in d.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            stats = self.get_stats(key)
            super().log_tabular(key if average_only else key + "/avg", stats[0])
            if not average_only:
                super().log_tabular(key + "/std", stats[1])
            if with_min_and_max:
                super().log_tabular(key + "/max", stats[3])
                super().log_tabular(key + "/min", stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict.get(key)
        if not v:
            return [np.nan, np.nan, np.nan, np.nan]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return [np.mean(vals), np.std(vals), np.min(vals), np.max(vals)]


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
