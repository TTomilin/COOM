import os
import time
from collections import deque
from typing import Dict, Tuple

import gymnasium as gym
import imageio
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.cpo_learner import CPOLearner
from sample_factory.algo.learning.ppo_detached_learner import PPODetachedLearner
from sample_factory.algo.learning.ppo_learner import PPOLearner
from sample_factory.algo.learning.ppo_lag_learner import PPOLagLearner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log


def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)


def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
    render_start = time.time()

    if cfg.save_video:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start


def save_video(frames, file_path):
    writer = imageio.get_writer(file_path, fps=35)  # ViZDoom runs at 35 fps
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = True

    train_dir = cfg.train_dir
    if cfg.timestamp:
        cfg = load_from_checkpoint(cfg)
    cfg.train_dir = train_dir  # Ensure we are running on the correct machine

    env_initials = ''.join(word[0].upper() for word in cfg.env.split('_'))
    video_file_path = f"{experiment_dir(cfg)}/{env_initials}_L{cfg.level}_{cfg.algo}_{cfg.resolution_eval}.mp4"
    if os.path.exists(video_file_path) and cfg.save_video and not cfg.overwrite_video:
        print(f"Video for {video_file_path} already exists. Skipping...")
        return ExperimentStatus.SUCCESS, 0

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (cfg.env_frameskip % eval_env_frameskip == 0), \
        f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    cfg.resolution = cfg.resolution_eval
    env_render = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=1), render_mode=render_mode
    )
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False
        env_render.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    if cfg.algo == 'PPOLag':
        learner_cls = PPOLagLearner
    elif cfg.algo == 'CPO':
        learner_cls = CPOLearner
    elif not cfg.actor_critic_share_weights:
        learner_cls = PPODetachedLearner
    else:
        learner_cls = PPOLearner

    if cfg.timestamp:
        checkpoints = learner_cls.get_checkpoints(learner_cls.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = learner_cls.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    episode_costs = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    reward_list = []

    obs, infos = env.reset()
    _, _ = env_render.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    episode_cost = 0
    finished_episode = [False for _ in range(env.num_agents)]

    video_frames = []
    num_episodes = 0

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            for _ in range(render_action_repeat):
                # last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                last_render_start = render_frame(cfg, env_render, video_frames, num_episodes, last_render_start)

                obs, rew, terminated, truncated, infos = env.step(actions)
                _, _, _, _, _ = env_render.step(actions)
                cost = infos[0]['cost']
                dones = make_dones(terminated, truncated)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                episode_cost += cost

                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug(f"Num frames {num_frames}...")

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)
                        episode_costs[agent_i].append(episode_cost)

                        if verbose:
                            log.info(
                                "Episode finished for agent %d at %d frames. Reward: %.3f, cost: %.3f",
                                agent_i,
                                num_frames,
                                episode_reward[agent_i],
                                episode_cost,
                            )
                        rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0
                        episode_cost = 0

                        if cfg.use_record_episode_statistics:
                            # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
                            if "episode" in infos[agent_i].keys():
                                num_episodes += 1
                                reward_list.append(infos[agent_i]["episode"]["r"])
                        else:
                            num_episodes += 1

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    # render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    render_frame(cfg, env_render, video_frames, num_episodes, last_render_start)
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_cost_str = "", ""
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_cost = np.mean(episode_costs[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ", "
                            avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
                        if not np.isnan(avg_cost):
                            if avg_cost_str:
                                avg_cost_str += ", "
                            avg_cost_str += f"#{agent_i}: {avg_cost:.3f}"

                    log.info(
                        "Avg episode reward: %.3f, avg cost: %.3f",
                        np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
                        np.mean([np.mean(episode_costs[i]) for i in range(env.num_agents)]),
                    )

            if num_episodes >= cfg.max_num_episodes:
                break

    env.close()

    # Save the video if frames were collected
    if cfg.save_video and video_frames:
        save_video(video_frames, video_file_path)
        log.info(f"Video saved to {video_file_path}")

    return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
        [len(episode_rewards[i]) for i in range(env.num_agents)]
    )
