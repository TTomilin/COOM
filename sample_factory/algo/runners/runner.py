from __future__ import annotations

import fcntl
import glob
import json
import math
import os
import re
import shutil
import time
from collections import OrderedDict, deque
from datetime import datetime
from io import BytesIO
from os.path import isdir, join
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import wandb
from PIL import Image
from matplotlib import pyplot as plt
from signal_slot.signal_slot import EventLoop, EventLoopObject, EventLoopStatus, Timer, process_name, signal
from tensorboardX import SummaryWriter

from sample_factory.algo.learning.batcher import Batcher
from sample_factory.algo.learning.learner_worker import LearnerWorker
from sample_factory.algo.sampling.sampler import AbstractSampler
from sample_factory.algo.sampling.stats import samples_stats_handler, stats_msg_handler, timing_msg_handler
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import (
    EPISODIC,
    LEARNER_ENV_STEPS,
    SAMPLES_COLLECTED,
    STATS_KEY,
    TIMING_STATS,
    TRAIN_STATS,
    ExperimentStatus,
)
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.cfg.arguments import cfg_dict, cfg_str, preprocess_cfg
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.gpu_utils import set_global_cuda_envvars
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import PolicyID, StatusCode
from sample_factory.utils.utils import (
    cfg_file,
    debug_log_every_n,
    ensure_dir_exists,
    experiment_dir,
    init_file_logger,
    log,
    memory_consumption_mb,
    save_git_diff,
    summaries_dir,
    frames_dir,
)
from sample_factory.utils.wandb_utils import init_wandb


class AlgoObserver:
    def on_init(self, runner: Runner) -> None:
        """Called after ctor, but before signal-slots are connected or any processes are started."""
        pass

    def on_connect_components(self, runner: Runner) -> None:
        """Connect additional signal-slot pairs in the observers if needed."""
        pass

    def on_start(self, runner: Runner) -> None:
        """Called right after sampling/learning processes are started."""
        pass

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        """Called after each training step."""
        pass

    def extra_summaries(self, runner: Runner, policy_id: PolicyID, env_steps: int, writer: SummaryWriter) -> None:
        pass

    def on_stop(self, runner: Runner) -> None:
        pass


MsgHandler = Callable[[Any, dict], None]
PolicyMsgHandler = Callable[[Any, dict, PolicyID], None]


class Runner(EventLoopObject, Configurable):
    def __init__(self, cfg, unique_name=None):
        Configurable.__init__(self, cfg)

        unique_name = Runner.__name__ if unique_name is None else unique_name
        self.event_loop: EventLoop = EventLoop(unique_loop_name=f"{unique_name}_EvtLoop", serial_mode=cfg.serial_mode)
        self.event_loop.owner = self
        EventLoopObject.__init__(self, self.event_loop, object_id=unique_name)

        self.status: StatusCode = ExperimentStatus.SUCCESS
        self.stopped: bool = False

        self.env_info: Optional[EnvInfo] = None

        self.reward_shaping: List[Optional[Dict]] = [None for _ in range(self.cfg.num_policies)]

        self.buffer_mgr = None

        self.learners: Dict[PolicyID, LearnerWorker] = dict()
        self.batchers: Dict[PolicyID, Batcher] = dict()
        self.sampler: Optional[AbstractSampler] = None

        self.timing = Timing("Runner profile")
        self.cfg.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # env_steps counts total number of simulation steps per policy (including frameskipped)
        self.env_steps: Dict[PolicyID, int] = dict()

        # samples_collected counts the total number of observations processed by the algorithm
        self.samples_collected = [0 for _ in range(self.cfg.num_policies)]

        # plotting visited locations as a 2D heatmap
        self.cumulative_heatmap = None
        self.map_img = None
        self.frames_dir = None
        self.last_heatmap_log = 0  # Step at which the last heatmap overlay was logged
        self.last_gif_log = 0  # Step at which the last GIF was logged

        self.total_env_steps_since_resume: Optional[int] = None
        self.start_time: float = time.time()

        # currently, this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()
        self.last_logged_step = -1

        self.report_interval_sec = 5.0
        self.avg_stats_intervals = (2, 12, 60)  # by default: 10 seconds, 60 seconds, 5 minutes
        self.summaries_interval_sec = self.cfg.experiment_summaries_interval  # sec
        self.heartbeat_report_sec = self.cfg.heartbeat_reporting_interval
        self.update_training_info_every_sec = 5.0

        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))
        self.throughput_stats = [deque([], maxlen=10) for _ in range(self.cfg.num_policies)]

        self.stats = dict()  # regular (non-averaged) stats
        self.avg_stats = dict()

        self.policy_avg_stats: Dict[str, List[Deque]] = dict()
        self.policy_lag = [dict() for _ in range(self.cfg.num_policies)]

        self._handle_restart()

        init_wandb(self.cfg)  # should be done before writers are initialized

        self.writers: Dict[int, SummaryWriter] = dict()
        for policy_id in range(self.cfg.num_policies):
            summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg)), str(policy_id))
            summary_dir = ensure_dir_exists(summary_dir)
            self.writers[policy_id] = SummaryWriter(summary_dir, flush_secs=cfg.flush_summaries_interval)

        # global msg handlers for messages from algo components
        self.msg_handlers: Dict[str, List[MsgHandler]] = {
            TIMING_STATS: [timing_msg_handler],
            STATS_KEY: [stats_msg_handler],
        }

        # handlers for policy-specific messages
        self.policy_msg_handlers: Dict[str, List[PolicyMsgHandler]] = {
            LEARNER_ENV_STEPS: [self._learner_steps_handler],
            EPISODIC: [self._episodic_stats_handler],
            TRAIN_STATS: [self._train_stats_handler],
            SAMPLES_COLLECTED: [samples_stats_handler],
        }

        self.observers: List[AlgoObserver] = []

        self.timers: List[Timer] = []

        def periodic(period, cb):
            t = Timer(self.event_loop, period)
            t.timeout.connect(cb)
            self.timers.append(t)

        periodic(self.report_interval_sec, self._update_stats_and_print_report)
        periodic(self.summaries_interval_sec, self._report_experiment_summaries)

        periodic(self.cfg.save_every_sec, self._save_policy)
        periodic(self.cfg.save_best_every_sec, self._save_best_policy)

        periodic(self.update_training_info_every_sec, self._propagate_training_info)

        if self.cfg.save_milestones_sec > 0:
            periodic(self.cfg.save_milestones_sec, self._save_milestone_policy)

        periodic(self.heartbeat_report_sec, self._check_heartbeat)

        self.heartbeat_dict = {}
        self.queue_size_dict = {}

        self.components_to_stop: List[EventLoopObject] = []
        self.component_profiles: Dict[str, Timing] = dict()

    # signals emitted by the runner
    @signal
    def save_periodic(self):
        ...

    @signal
    def save_best(self):
        ...

    @signal
    def update_training_info(self):
        ...

    @signal
    def save_milestone(self):
        ...

    @signal
    def stop(self):
        """Emitted when we're about to stop the experiment."""
        ...

    @signal
    def all_components_stopped(self):
        ...

    def _handle_restart(self):
        exp_dir = experiment_dir(self.cfg, mkdir=False)
        if isdir(exp_dir):
            log.debug(f"Experiment dir {exp_dir} already exists!")
            if self.cfg.restart_behavior == "resume":
                log.debug(f"Resuming existing experiment from {exp_dir}...")
            else:
                if self.cfg.restart_behavior == "restart":
                    attempt = 0
                    old_exp_dir = exp_dir
                    while isdir(old_exp_dir):
                        attempt += 1
                        old_exp_dir = f"{exp_dir}_old{attempt:04d}"

                    # move the existing experiment dir to a new one with a suffix
                    log.debug(f"Moving the existing experiment dir to {old_exp_dir}...")
                    shutil.move(exp_dir, old_exp_dir)
                elif self.cfg.restart_behavior == "overwrite":
                    log.debug(f"Overwriting the existing experiment dir {exp_dir}...")
                    shutil.rmtree(exp_dir)
                else:
                    raise ValueError(f"Unknown restart behavior {self.cfg.restart_behavior}")

                log.debug(f"Starting training in {exp_dir}...")

    def _process_msg(self, msgs):
        if isinstance(msgs, (dict, OrderedDict)):
            msgs = (msgs,)

        if not (isinstance(msgs, (List, Tuple)) and isinstance(msgs[0], (dict, OrderedDict))):
            log.error("While parsing a message: expected a dictionary or list/tuple of dictionaries, found %r", msgs)
            return

        for msg in msgs:
            # some messages are policy-specific
            policy_id = msg.get("policy_id", None)

            for key in msg:
                for handler in self.msg_handlers.get(key, ()):
                    handler(self, msg)
                if policy_id is not None:
                    for handler in self.policy_msg_handlers.get(key, ()):
                        handler(self, msg, policy_id)

    @staticmethod
    def _learner_steps_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        env_steps: int = msg[LEARNER_ENV_STEPS]
        if policy_id in runner.env_steps:
            delta = env_steps - runner.env_steps[policy_id]
            runner.total_env_steps_since_resume += delta
        elif runner.total_env_steps_since_resume is None:
            runner.total_env_steps_since_resume = 0

        runner.env_steps[policy_id] = env_steps

    @staticmethod
    def _episodic_stats_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        s = msg[EPISODIC]
        for _, key, value in iterate_recursively(s):
            if key not in runner.policy_avg_stats:
                max_len = runner.cfg.heatmap_avg if key == 'heatmap' else runner.cfg.stats_avg
                runner.policy_avg_stats[key] = [
                    deque(maxlen=max_len) for _ in range(runner.cfg.num_policies)
                ]

            if isinstance(value, np.ndarray) and value.ndim > 0 and key != 'heatmap':
                if len(value) > runner.policy_avg_stats[key][policy_id].maxlen:
                    # increase maxlen to make sure we never ignore any stats from the environments
                    runner.policy_avg_stats[key][policy_id] = deque(maxlen=len(value))

                runner.policy_avg_stats[key][policy_id].extend(value)
            else:
                runner.policy_avg_stats[key][policy_id].append(value)

    @staticmethod
    def _train_stats_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        """We write the train summaries to disk right away instead of accumulating them."""
        train_stats = msg[TRAIN_STATS]
        for key, scalar in train_stats.items():
            runner.writers[policy_id].add_scalar(f"train/{key}", scalar, runner.env_steps[policy_id])

        for key in ["version_diff_min", "version_diff_max", "version_diff_avg"]:
            if key in train_stats:
                runner.policy_lag[policy_id][key] = train_stats[key]

    def _get_perf_stats(self):
        # total env steps simulated across all policies
        fps_stats = []
        for avg_interval in self.avg_stats_intervals:
            fps_for_interval = math.nan
            if len(self.fps_stats) > 1:
                t1, x1 = self.fps_stats[max(0, len(self.fps_stats) - 1 - avg_interval)]
                t2, x2 = self.fps_stats[-1]
                fps_for_interval = (x2 - x1) / (t2 - t1)

            fps_stats.append(fps_for_interval)

        # learning throughput per policy (in observations per sec)
        sample_throughput = dict()
        for policy_id in range(self.cfg.num_policies):
            sample_throughput[policy_id] = math.nan
            if len(self.throughput_stats[policy_id]) > 1:
                t1, x1 = self.throughput_stats[policy_id][0]
                t2, x2 = self.throughput_stats[policy_id][-1]
                sample_throughput[policy_id] = (x2 - x1) / (t2 - t1)

        return fps_stats, sample_throughput

    def print_stats(self, fps, sample_throughput, total_env_steps):
        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(f"{int(interval * self.report_interval_sec)} sec: {fps_value:.1f}")
        fps_str = f'({", ".join(fps_str)})'

        samples_per_policy = ", ".join([f"{p}: {s:.1f}" for p, s in sample_throughput.items()])

        lag_stats = self.policy_lag[0]
        lag = AttrDict()
        for key in ["min", "avg", "max"]:
            lag[key] = lag_stats.get(f"version_diff_{key}", -1)
        policy_lag_str = f"min: {lag.min:.1f}, avg: {lag.avg:.1f}, max: {lag.max:.1f}"

        log.debug(
            "Fps is %s. Total num frames: %d. Throughput: %s. Samples: %d. Policy #0 lag: (%s)",
            fps_str,
            total_env_steps,
            samples_per_policy,
            sum(self.samples_collected),
            policy_lag_str,
        )

        if "reward" in self.policy_avg_stats:
            policy_reward_stats = []
            for policy_id in range(self.cfg.num_policies):
                reward_stats = self.policy_avg_stats["reward"][policy_id]
                if len(reward_stats) > 0:
                    policy_reward_stats.append((policy_id, f"{np.mean(reward_stats):.3f}"))
            log.debug("Avg episode reward: %r", policy_reward_stats)

        if "cost" in self.policy_avg_stats:
            policy_cost_stats = []
            for policy_id in range(self.cfg.num_policies):
                cost_stats = self.policy_avg_stats["cost"][policy_id]
                if len(cost_stats) > 0:
                    policy_cost_stats.append((policy_id, f"{np.mean(cost_stats):.3f}"))
            log.debug("Avg episode cost: %r", policy_cost_stats)

    def _update_stats_and_print_report(self):
        """
        Called periodically (every self.report_interval_sec seconds).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        # don't have enough statistic from the learners yet
        if len(self.env_steps) < self.cfg.num_policies:
            return

        if self.total_env_steps_since_resume is None:
            return

        now = time.time()
        self.fps_stats.append((now, self.total_env_steps_since_resume))

        for policy_id in range(self.cfg.num_policies):
            self.throughput_stats[policy_id].append((now, self.samples_collected[policy_id]))

        fps_stats, sample_throughput = self._get_perf_stats()
        total_env_steps = sum(self.env_steps.values())
        self.print_stats(fps_stats, sample_throughput, total_env_steps)

    def log_heatmap(self, heatmap: np.ndarray, global_step: int, tag: str):
        # Transpose the heatmap
        heatmap = np.flipud(heatmap.T)

        # Determine aspect ratio of the histogram
        height, width = heatmap.shape
        aspect_ratio = width / height

        # Define additional space for the colorbar
        colorbar_width_factor = 0.25  # Approximation of colorbar width to figure width

        # Calculate figure dimensions basing the width on a fixed height
        base_height = 2 if self.env_info.name in ['precipice_plunge', 'detonators_dilemma'] else 7
        fig_width = base_height * aspect_ratio * (1 + colorbar_width_factor)

        # Create a BytesIO buffer to save image
        buf = BytesIO()
        plt.figure(figsize=(fig_width, base_height))
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.savefig(buf, format='png')
        # plt.show()
        plt.close()

        buf.seek(0)
        image = Image.open(buf)
        wandb.log({tag: wandb.Image(image)}, step=global_step)

    def log_overlay(self, heatmap: np.ndarray, global_step: int, tag: str):
        # Transpose the heatmap
        heatmap = np.flipud(heatmap.T)

        # Determine aspect ratio of the histogram
        height, width = heatmap.shape
        aspect_ratio = width / height

        # Define additional space for the colorbar
        colorbar_width_factor = 0.25  # Approximation of colorbar width to figure width

        # Calculate figure dimensions basing the width on a fixed height
        base_height = 2 if self.env_info.name in ['precipice_plunge', 'detonators_dilemma'] else 4
        fig_width = base_height * aspect_ratio * (1 + colorbar_width_factor)

        # Create a figure and axis to plot the map and heatmap
        fig, ax = plt.subplots(figsize=(fig_width, base_height))

        # Display the map
        ax.imshow(self.map_img, extent=[0, width, 0, height])

        # Overlay the heatmap: adjust 'alpha' for transparency, cmap for the color map
        ax.imshow(heatmap, cmap='viridis', alpha=0.5, interpolation='nearest', extent=[0, width, 0, height])

        # Remove x and y ticks as they are meaningless here
        plt.xticks([])
        plt.yticks([])

        # Add a step counter on the frame
        plt.text(0.99, 0.99, f'Step: {global_step:09d}', fontsize=12, color='white',
                 horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)

        # Save the plot to a buffer
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        if not self.frames_dir:
            self.frames_dir = join(frames_dir(experiment_dir(cfg=self.cfg)))  # Create a directory for storing frames
        frame_path = os.path.join(self.frames_dir, f"frame_{global_step:09d}.png")
        plt.savefig(frame_path, format='png')
        plt.show()
        plt.close()

        # Log to Weights & Biases
        image = Image.open(buf)

        wandb.log({tag: wandb.Image(image)}, step=global_step)

    def create_and_upload_gif(self, tag):
        # List all the frames, sorted by extracted step number
        frame_files = sorted(
            glob.glob(os.path.join(self.frames_dir, "frame_*.png")),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
        )
        frames = [Image.open(frame) for frame in frame_files]

        # Create a BytesIO buffer to hold the GIF
        gif_buffer = BytesIO()

        if frames:
            # Total GIF duration in seconds
            total_duration_secs = self.cfg.gif_duration

            # Calculate the duration each frame should be displayed to fit the total
            frame_duration = int((total_duration_secs * 1000) / len(frames))  # Convert seconds to milliseconds

            # Enforce minimum and maximum duration limits
            frame_duration = max(10, min(frame_duration, 100))

            # Create GIF in the buffer
            frames[0].save(
                gif_buffer, format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=frame_duration, loop=0
            )
            gif_buffer.seek(0)  # Rewind to the start of the GIF buffer

            # Log the GIF to wandb
            wandb.log({tag: wandb.Video(gif_buffer, format="gif")})
            print(f"{total_duration_secs} second GIF uploaded at {self.last_gif_log} steps to wandb as {tag}")

            # Clear the buffer if no longer needed
            gif_buffer.close()

    def _report_experiment_summaries(self):
        memory_mb = memory_consumption_mb()

        fps_stats, sample_throughput = self._get_perf_stats()
        fps = fps_stats[0]

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            writer = self.writers[policy_id]
            if policy_id == default_policy:
                if not math.isnan(fps):
                    writer.add_scalar("perf/_fps", fps, env_steps)

                writer.add_scalar("stats/master_process_memory_mb", float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and self.total_train_seconds > 300):
                        writer.add_scalar(f"stats/{key}", np.mean(value), env_steps)

                for key, value in self.stats.items():
                    writer.add_scalar(f"stats/{key}", value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                writer.add_scalar("perf/_sample_throughput", sample_throughput[policy_id], env_steps)

            if self.cfg.with_wandb and 'heatmap' in self.policy_avg_stats:
                heatmap = np.mean(self.policy_avg_stats['heatmap'], axis=1).squeeze()
                if self.cumulative_heatmap is None:
                    # Initialize the cumulative heatmap to the correct shape
                    self.cumulative_heatmap = np.zeros_like(heatmap)

                # Accumulate the heatmap data
                self.cumulative_heatmap += heatmap
                if env_steps - self.last_heatmap_log >= self.cfg.heatmap_log_interval:
                    if self.cfg.log_overlay:
                        self.log_overlay(heatmap, env_steps, 'traversal/overlay')
                    if self.cfg.log_heatmap:
                        self.log_heatmap(self.cumulative_heatmap, env_steps, "traversal/cumulative")
                        self.log_heatmap(heatmap, env_steps, "traversal/window")
                    self.last_heatmap_log = env_steps

                if env_steps - self.last_gif_log >= self.cfg.gif_log_interval:
                    self.last_gif_log = env_steps
                    self.create_and_upload_gif("traversal/evolution")

            for key, stat in self.policy_avg_stats.items():
                if key == 'heatmap':
                    continue  # Skip if it's histogram, already logged
                if len(stat[policy_id]) >= stat[policy_id].maxlen or (
                        len(stat[policy_id]) > 10 and self.total_train_seconds > 300
                ):
                    stat_value = np.mean(stat[policy_id])

                    if "/" in key:
                        # custom summaries have their own sections in tensorboard
                        avg_tag = key
                        min_tag = f"{key}_min"
                        max_tag = f"{key}_max"
                    elif key in ("reward", "len"):
                        # reward and length get special treatment
                        avg_tag = f"{key}/{key}"
                        min_tag = f"{key}/{key}_min"
                        max_tag = f"{key}/{key}_max"
                    else:
                        avg_tag = f"policy_stats/avg_{key}"
                        min_tag = f"policy_stats/avg_{key}_min"
                        max_tag = f"policy_stats/avg_{key}_max"

                    writer.add_scalar(avg_tag, float(stat_value), env_steps)

                    # for key stats report min/max as well
                    if key in ("reward", "cost", "true_objective", "len"):
                        writer.add_scalar(min_tag, float(min(stat[policy_id])), env_steps)
                        writer.add_scalar(max_tag, float(max(stat[policy_id])), env_steps)

            self._observers_call(AlgoObserver.extra_summaries, self, policy_id, writer, env_steps)

        # Video logging
        if self.cfg.with_wandb:
            self.log_new_videos_to_wandb()
            # Ensure to flush/write all accumulated wandb logs
            wandb.log({})

        for w in self.writers.values():
            w.flush()

    def log_new_videos_to_wandb(self):
        # List all video files in the directory
        video_dir = join(experiment_dir(cfg=self.cfg), self.cfg.video_dir)
        # Ensure the directory exists
        if not os.path.exists(video_dir):
            return
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

        # Filter files based on episode number being greater than last_logged_episode
        episode_pattern = re.compile(r"doom-step-(\d+).mp4")
        new_videos = []
        for video_file in video_files:
            match = episode_pattern.search(video_file)
            if match:
                step_number = int(match.group(1))
                if step_number > self.last_logged_step:
                    new_videos.append((step_number, video_file))

        def is_file_locked(file_path):
            """Check if the file is locked by another process."""
            try:
                with open(file_path, 'rb') as f:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f, fcntl.LOCK_UN)
                return False  # File is not locked
            except IOError:
                return True  # File is locked

        # Log new videos to wandb
        for step_number, video_file in sorted(new_videos):
            video_path = join(video_dir, video_file)
            video_tag = f"video/step_{step_number}"

            if not is_file_locked(video_path):
                # Video has been stored and is ready to be uploaded
                wandb.log({video_tag: wandb.Video(video_path, format='mp4')})

                # Now that we've successfully logged it, update last_logged_step
                self.last_logged_step = max(self.last_logged_step, step_number)
                log.info(f"Logged video at step {step_number} to wandb as {video_tag}. New last_logged_step: {self.last_logged_step}")
            else:
                log.warning(f"Video at step {step_number} is locked and cannot be uploaded to wandb. Last logged step: {self.last_logged_step}")


    def _propagate_training_info(self):
        """
        Send the training stats (such as the number of processed env steps) to the sampler.
        This can be used by the envs to configure curriculums and so on.
        """

        training_info: Dict[PolicyID, Dict[str, Any]] = dict()
        for policy_id in range(self.cfg.num_policies):
            training_info[policy_id] = dict(
                policy_id=policy_id,
                # "approx" here because it will lag behind a little bit due to the async nature of the system
                approx_total_training_steps=self.env_steps.get(policy_id, 0),
                reward_shaping=self.reward_shaping[policy_id],
                # add more stats if needed (commented by default for efficiency)
                # stats=self.stats,
                # avg_stats=self.avg_stats,
                # policy_avg_stats=self.policy_avg_stats,
            )

        self.update_training_info.emit(training_info)

    def update_reward_shaping(self, policy_id: PolicyID, reward_shaping: Dict[str, Any]) -> None:
        self.reward_shaping[policy_id] = reward_shaping

        # send the updated data to other components (e.g. the sampler)
        # this allows us to change reward shaping on the fly, PBT can take advantage of this
        self._propagate_training_info()

    def _save_policy(self):
        self.save_periodic.emit()

    def _save_milestone_policy(self):
        self.save_milestone.emit()

    def _save_best_policy(self):
        # don't have enough statistic from the learners yet
        if len(self.env_steps) < self.cfg.num_policies:
            return

        metric = self.cfg.save_best_metric
        if metric in self.policy_avg_stats:
            for policy_id in range(self.cfg.num_policies):
                # check if number of samples collected is greater than cfg.save_best_after
                env_steps = self.env_steps[policy_id]
                if env_steps < self.cfg.save_best_after:
                    continue

                stats = self.policy_avg_stats[metric][policy_id]
                if len(stats) > 0:
                    avg_metric = np.mean(stats)
                    self.save_best.emit(policy_id, metric, avg_metric)

    @staticmethod
    def _register_msg_handler(handlers_dict, key, func):
        handlers_dict[key] = func

    def register_msg_handler(self, key, func):
        self._register_msg_handler(self.msg_handlers, key, func)

    def register_policy_msg_handler(self, key, func):
        self._register_msg_handler(self.policy_msg_handlers, key, func)

    def register_episodic_stats_handler(self, func: PolicyMsgHandler):
        self.policy_msg_handlers[EPISODIC].append(func)

    def register_observer(self, observer: AlgoObserver) -> None:
        self.observers.append(observer)

    def _observers_call(self, func, *args, **kwargs) -> None:
        for observer in self.observers:
            getattr(observer, func.__name__)(*args, **kwargs)

    def _save_cfg(self):
        fname = cfg_file(self.cfg)
        with open(fname, "w") as json_file:
            log.debug(f"Saving configuration to {fname}...")
            json.dump(cfg_dict(self.cfg), json_file, indent=2)

    def _make_batcher(self, event_loop, policy_id: PolicyID):
        return Batcher(event_loop, policy_id, self.buffer_mgr, self.cfg, self.env_info)

    def _make_learner(self, event_loop, policy_id: PolicyID, batcher: Batcher):
        return LearnerWorker(
            event_loop,
            self.cfg,
            self.env_info,
            self.buffer_mgr,
            batcher,
            policy_id=policy_id,
        )

    def _make_sampler(self, sampler_cls: type, event_loop: EventLoop):
        assert len(self.learners) == self.cfg.num_policies, "Learners not created yet"
        param_servers = {policy: self.learners[policy].param_server for policy in self.learners}
        return sampler_cls(event_loop, self.buffer_mgr, param_servers, self.cfg, self.env_info)

    def init(self) -> StatusCode:
        set_global_cuda_envvars(self.cfg)
        self.env_info = obtain_env_info_in_a_separate_process(self.cfg)

        # Load the map image
        base_path = Path(__file__).parent.parent.parent.resolve()
        map_path = join(base_path, "doom", "env", "scenarios", f"{self.env_info.name}.png")
        self.map_img = cv2.imread(map_path)

        # # Check if the image was loaded successfully
        # if self.map_img is None:
        #     raise FileNotFoundError(f"Map image not found at path: {map_path}")
        #
        # # Define the desired width or height while maintaining aspect ratio
        # # For example, set the width to 800 pixels
        # desired_width = 800
        #
        # # Get current dimensions
        # original_height, original_width = self.map_img.shape[:2]
        # aspect_ratio = original_width / original_height
        #
        # # Calculate the new height to maintain aspect ratio
        # desired_height = int(desired_width / aspect_ratio)
        #
        # # Resize the image
        # self.map_img = cv2.resize(self.map_img, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
        #
        # # Optional: Verify the new size
        # print(f"Resized map image to: {self.map_img.shape[1]}x{self.map_img.shape[0]} pixels")


        for policy_id in range(self.cfg.num_policies):
            self.reward_shaping[policy_id] = self.env_info.reward_shaping_scheme

        # check for any incompatible arguments
        if not preprocess_cfg(self.cfg, self.env_info):
            return ExperimentStatus.FAILURE

        log.debug(f"Starting experiment with the following configuration:\n{cfg_str(self.cfg)}")

        init_file_logger(self.cfg)
        self._save_cfg()
        save_git_diff(experiment_dir(self.cfg))

        self.buffer_mgr = BufferMgr(self.cfg, self.env_info)

        self._observers_call(AlgoObserver.on_init, self)

        return ExperimentStatus.SUCCESS

    def _on_start(self):
        """Override this in a subclass to do something right when the experiment is started."""
        self.sampler.init()
        self._propagate_training_info()
        self._observers_call(AlgoObserver.on_start, self)

    def _setup_component_heartbeat(self, component: HeartbeatStoppableEventLoopObject):
        """
        Groups components with heartbeat mechanism by type and records starting time.
        When all components of the same type do not respond in the reporting timeframe, stops the run
        """
        component_type = type(component)
        if component_type not in self.heartbeat_dict:
            self.heartbeat_dict[component_type] = {}
        type_dict = self.heartbeat_dict[component_type]
        type_dict[component.object_id] = None

        # setup up queue_size report with heartbeat, grouped by event_loop_process_name
        p_name = process_name(component.event_loop.process)
        if p_name not in self.queue_size_dict:
            self.queue_size_dict[p_name] = 0

        component.heartbeat.connect(self._receive_heartbeat)

    def _receive_heartbeat(self, component_type: type, component_id: str, p_name: str, qsize: int):
        """
        Record the time the most recent heartbeat was received
        """
        curr_time = time.time()
        heartbeat_time = self.heartbeat_dict[component_type][component_id]
        if heartbeat_time is None:
            log.info(f"Heartbeat connected on {component_id}")
        elif curr_time - heartbeat_time > self.heartbeat_report_sec:
            log.info(f"Heartbeat reconnected after {int(curr_time - heartbeat_time)} seconds from {component_id}")
        self.heartbeat_dict[component_type][component_id] = curr_time
        self.queue_size_dict[p_name] = qsize

    def _check_heartbeat(self):
        """
        Reports components whose last heartbeat signal is longer than self.heartbeat_report_sec.
        If all components of the same type fail, stop the run
        """
        curr_time = time.time()
        comp_list = []
        type_list = []
        none_list = []
        for component_type, heartbeat_dict in self.heartbeat_dict.items():
            num_components = len(heartbeat_dict)
            num_stopped = 0
            for component_id, heartbeat_time in heartbeat_dict.items():
                if heartbeat_time is None:
                    none_list.append(component_id)
                    continue
                if curr_time - heartbeat_time > self.heartbeat_report_sec:
                    comp_list.append(f"{component_id} ({(int(curr_time - heartbeat_time))} seconds)")
                    num_stopped += 1
            if num_stopped == num_components:
                type_list.append(str(component_type))

        if len(none_list) > 0:
            wait_time = time.time() - self.start_time
            log.debug(f"Components not started: {', '.join(none_list)}, {wait_time=:.1f} seconds")
            if wait_time > 3 * self.heartbeat_report_sec:
                log.error(f"Components take too long to start: {', '.join(none_list)}. Aborting the experiment!\n\n\n")
                self._stop_training(failed=True)

        if len(comp_list) > 0:
            log.error(f"No heartbeat for components: {', '.join(comp_list)}")

        if len(type_list) > 0:
            log.error(f"Stopping training due to lack of heartbeats from {', '.join(type_list)}")
            self._stop_training(failed=True)

        for p_name, qsize in self.queue_size_dict.items():
            if qsize > 5:
                debug_log_every_n(1000, f"Process: {p_name} has queue size: {qsize}")

    def _setup_component_termination(self, stop_signal: signal, component_to_stop: HeartbeatStoppableEventLoopObject):
        stop_signal.connect(component_to_stop.on_stop)
        self.components_to_stop.append(component_to_stop)
        component_to_stop.stop.connect(self._component_stopped)

    def connect_components(self):
        self.event_loop.start.connect(self._on_start)

        sampler = self.sampler
        for policy_id in range(self.cfg.num_policies):
            # when runner is ready we initialize the learner first and then all other components in a chain
            learner_worker = self.learners[policy_id]
            batcher = self.batchers[policy_id]
            self.event_loop.start.connect(learner_worker.init)
            learner_worker.initialized.connect(batcher.init)
            sampler.connect_model_initialized(policy_id, learner_worker.model_initialized)

            # key connections - sampler and batcher exchanging connections back and forth
            sampler.connect_on_new_trajectories(policy_id, batcher.on_new_trajectories)
            sampler.connect_trajectory_buffers_available(batcher.trajectory_buffers_available)

            # batcher gives learner batches of trajectories ready for learning
            batcher.training_batches_available.connect(learner_worker.on_new_training_batch)
            # once learner is done with a training batch, it is given back to the batcher
            learner_worker.training_batch_released.connect(batcher.on_training_batch_released)

            # signals that allow us to throttle the sampler if the learner can't keep up
            sampler.connect_stop_experience_collection(batcher.stop_experience_collection)
            sampler.connect_resume_experience_collection(batcher.resume_experience_collection)

            # auxiliary connections, such as summary reporting and checkpointing
            learner_worker.finished_training_iteration.connect(self._after_training_iteration)
            learner_worker.report_msg.connect(self._process_msg)
            sampler.connect_report_msg(self._process_msg)
            sampler.connect_update_training_info(self.update_training_info)
            self.save_periodic.connect(learner_worker.save)
            self.save_best.connect(learner_worker.save_best)
            self.save_milestone.connect(learner_worker.save_milestone)

            # stop components when needed
            self._setup_component_termination(self.stop, batcher)
            self._setup_component_termination(batcher.stop, learner_worker)

            # Heartbeats
            self._setup_component_heartbeat(batcher)
            self._setup_component_heartbeat(learner_worker)

        for sampler_component in sampler.stoppable_components():
            self._setup_component_termination(self.stop, sampler_component)

        for sampler_component in sampler.heartbeat_components():
            self._setup_component_heartbeat(sampler_component)

        # final cleanup
        self.all_components_stopped.connect(self._on_everything_stopped)

        # connect additional signal-slot pairs in the observers if needed
        self._observers_call(AlgoObserver.on_connect_components, self)

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds
        return end

    def _after_training_iteration(self, training_iteration_since_resume: int):
        self._observers_call(AlgoObserver.on_training_step, self, training_iteration_since_resume)

        if self._should_end_training():
            self._stop_training()

    def _stop_training(self, failed: bool = False) -> None:
        if not self.stopped:
            self._propagate_training_info()

            self._observers_call(AlgoObserver.on_stop, self)

            self._save_policy()
            self._save_best_policy()

            for timer in self.timers:
                timer.stop()
            self.stop.emit(self.object_id)

            if failed:
                self.status = ExperimentStatus.FAILURE

            self.stopped = True

    def _component_stopped(self, component_obj_id, component_profiles: Dict[str, Timing]):
        remaining = []
        for i, component in enumerate(self.components_to_stop):
            if component.object_id == component_obj_id:
                log.debug(f"Component {component_obj_id} stopped!")
                continue

            try:
                if not component.event_loop.process.is_alive():
                    log.warning(f"Component {component.object_id} process died already! Don't wait for it.")
                    continue
            except AttributeError:
                # in serial mode there's no process, plus event_loop can be None for stopped components
                pass

            remaining.append(component)

        self.components_to_stop = remaining
        if self.components_to_stop and self.status == ExperimentStatus.FAILURE:
            log.debug(f"Waiting for {[c.object_id for c in self.components_to_stop]} to stop...")

        self.component_profiles.update(component_profiles)

        if not self.components_to_stop:
            self.all_components_stopped.emit()

    def _on_everything_stopped(self):
        # sort profiles by name
        self.component_profiles = sorted(list(self.component_profiles.items()), key=lambda x: x[0])
        for component, profile in self.component_profiles:
            log.info(profile)

        for w in self.writers.values():
            w.flush()

        assert self.event_loop.owner is self
        self.event_loop.stop()

    # noinspection PyBroadException
    def run(self) -> StatusCode:
        with self.timing.timeit("main_loop"):
            try:
                evt_loop_status = self.event_loop.exec()
                self.status = (
                    ExperimentStatus.INTERRUPTED if evt_loop_status == EventLoopStatus.INTERRUPTED else self.status
                )
                self.stop.emit(self.object_id)
            except Exception:
                log.exception(f"Uncaught exception in {self.object_id} evt loop")
                self.status = ExperimentStatus.FAILURE

        log.info(self.timing)
        if self.total_env_steps_since_resume is None:
            self.total_env_steps_since_resume = 0
        fps = self.total_env_steps_since_resume / self.timing.main_loop
        log.info("Collected %r, FPS: %.1f", self.env_steps, fps)
        return self.status
