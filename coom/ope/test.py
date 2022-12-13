from multiprocessing import cpu_count, Pool

import argparse
import chainer
import cv2
import gym
import gzip
import imageio
import numpy as np
import os
import time
import traceback
from pathlib import Path

from lib.constants import DOOM_GAMES
from lib.utils import log, mkdir, pre_process_image_tensor, post_process_image_tensor

try:
    from env.wrappers import ViZDoomWrapper
except Exception as e:
    None
from model import MDN_RNN
from vision import CVAE
from MC_auxiliary import action, transform_to_weights

ID = "test"


def worker(worker_arg_tuple):
    try:
        rollout_num, args, vision, model, W_c, b_c, output_dir = worker_arg_tuple

        np.random.seed()

        model.reset_state()

        render_mode = 'rgb_array' if args.record else 'human'

        if args.game in DOOM_GAMES:
            env = ViZDoomWrapper(args.game)
        else:
            env = gym.make(args.game, render_mode=render_mode, domain_randomize=args.domain_randomization)

        h_t = np.zeros(args.hidden_dim).astype(np.float32)
        c_t = np.zeros(args.hidden_dim).astype(np.float32)

        t = 0
        cumulative_reward = 0
        frames_array = []

        observation = env.reset()[0]

        start_time = time.time()
        while True:
            observation = cv2.resize(observation, (args.frame_resize, args.frame_resize))
            observation = pre_process_image_tensor(np.expand_dims(observation, 0))

            z_t = vision.encode(observation, return_z=True).data[0]

            a_t = action(args, W_c, b_c, z_t, h_t, c_t, None)

            observation, reward, done, _, _ = env.step(a_t)
            if args.record:
                frames_array.append(env.render())
            elif args.render:
                env.render()
            model(z_t, a_t, temperature=args.temperature)
            cumulative_reward += reward

            h_t = model.get_h().data[0]
            c_t = model.get_c().data[0]

            t += 1

            if done or t >= args.max_timesteps:
                break

        log(ID,
            "> Rollout #{} finished after {} timesteps in {:.2f}s with cumulative reward {:.2f}".format(
                (rollout_num + 1), t,
                (time.time() - start_time),
                cumulative_reward)
            )

        env.close()

        if args.record:
            frames_array = np.asarray(frames_array)
            imageio.mimsave(os.path.join(output_dir, str(rollout_num + 1) + '.gif'),
                            post_process_image_tensor(frames_array),
                            fps=20)

        return cumulative_reward
    except Exception:
        print(traceback.format_exc())
        return 0.


def main():
    parser = argparse.ArgumentParser(description='World Models ' + ID)
    parser.add_argument('--data_dir', '-d', default="data/wm", help='The base data/output directory')
    parser.add_argument('--game', default='CarRacing-v2',
                        help='Game to use')  # https://www.gymlibrary.dev/environments/box2d/car_racing/
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--rollouts', '-n', default=5, type=int, help='Number of times to rollout')
    parser.add_argument('--frame_resize', default=64, type=int, help='h x w resize of each observation frame')
    parser.add_argument('--hidden_dim', default=256, type=int, help='LSTM hidden units')
    parser.add_argument('--z_dim', '-z', default=32, type=int, help='dimension of encoded vector')
    parser.add_argument('--mixtures', default=5, type=int, help='number of gaussian mixtures for MDN')
    parser.add_argument('--temperature', '-t', default=1.0, type=float, help='Temperature (tau) for MDN-RNN (model)')
    parser.add_argument('--predict_done', action='store_true', help='Whether MDN-RNN should also predict done state')
    parser.add_argument('--max_timesteps', default=200, type=int, help='Max timesteps per rollout')
    parser.add_argument('--cores', default=0, type=int, help='Number of CPU cores to use. 0=all cores')
    parser.add_argument('--weights_type', default=1, type=int,
                        help="1=action_dim*(z_dim+hidden_dim), 2=z_dim+2*hidden_dim")
    parser.add_argument('--record', action='store_true', help='Record as gifs')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--domain_randomization', action='store_true', help='Randomize the colors of the environment')

    args = parser.parse_args()
    log(ID, "args =\n " + str(vars(args)).replace(",", ",\n "))

    env = ViZDoomWrapper(args.game) if args.game in DOOM_GAMES else gym.make(args.game)
    action_dim = len(env.action_space.low)
    args.action_dim = action_dim
    env = None

    if args.cores == 0:
        cores = cpu_count()
    else:
        cores = args.cores

    ope_dir = Path(__file__).parent.resolve()
    output_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, ID)
    mkdir(output_dir)
    model_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, 'model')
    vision_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, 'vision')
    controller_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, 'controller')

    model = MDN_RNN(args.hidden_dim, args.z_dim, args.mixtures, args.predict_done)
    chainer.serializers.load_npz(os.path.join(model_dir, "model.model"), model)
    vision = CVAE(args.z_dim)
    chainer.serializers.load_npz(os.path.join(vision_dir, "vision.model"), vision)
    # controller = np.random.randn(action_dim * (args.z_dim + args.hidden_dim) + action_dim).astype(np.float32)
    # controller = np.random.randn(args.z_dim + 2 * args.hidden_dim).astype(np.float32)
    controller = np.load(os.path.join(controller_dir, "controller.model"))['xmean']
    W_c, b_c = transform_to_weights(args, controller)

    log(ID, "Starting")

    worker_arg_tuples = []
    for rollout_num in range(args.rollouts):
        worker_arg_tuples.append((rollout_num, args, vision, model.copy(), W_c, b_c, output_dir))
    pool = Pool(cores)
    cumulative_rewards = pool.map(worker, worker_arg_tuples)
    pool.close()
    pool.join()

    log(ID, "Cumulative Rewards:")
    for rollout_num in range(args.rollouts):
        log(ID, "> #{} = {:.2f}".format((rollout_num + 1), cumulative_rewards[rollout_num]))

    log(ID, "Mean: {:.2f} Std: {:.2f}".format(np.mean(cumulative_rewards), np.std(cumulative_rewards)))
    log(ID, "Highest: #{} = {:.2f} Lowest: #{} = {:.2f}"
        .format(np.argmax(cumulative_rewards) + 1, np.amax(cumulative_rewards),
                np.argmin(cumulative_rewards) + 1, np.amin(cumulative_rewards)))

    cumulative_rewards_file = os.path.join(output_dir, "cumulative_rewards.npy.gz")
    log(ID, "Saving cumulative rewards to: " + os.path.join(output_dir, "cumulative_rewards.npy.gz"))
    with gzip.GzipFile(cumulative_rewards_file, "w") as file:
        np.save(file, cumulative_rewards)

    # To load:
    # with gzip.GzipFile(cumulative_rewards_file, "r") as file:
    #     cumulative_rewards = np.load(file)

    log(ID, "Done")


if __name__ == '__main__':
    main()
