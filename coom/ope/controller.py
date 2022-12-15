from multiprocessing import cpu_count, Pool

import argparse
import ast
import chainer
import cupy as cp
import cv2
import gym
import imageio
import math
import numpy as np
import os
import re
import socket
import time
import traceback
import wandb
from chainer.backends import cuda
from datetime import datetime
from io import BytesIO
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread, Lock, Event

from coom.utils.wandb import init_wandb
from lib.utils import log, mkdir, pre_process_image_tensor, post_process_image_tensor, reset_with_domain

try:
    from env.wrappers import ViZDoomWrapper
except Exception as e:
    None
from lib.constants import DOOM_GAMES
from model import MDN_RNN
from vision import CVAE
from lib.data import ModelDataset

from MC_auxiliary import transform_to_weights, action

ID = "controller"

CLUSTER_WORKERS = ['machine01', 'machine02', 'machine03', 'machine04', 'machine05', 'machine06',
                   'machine07', 'machine08', 'machine09', 'machine10', 'machine11', 'machine12']
CLUSTER_DISPATCHER = 'machine01'
CLUSTER_DISPATCHER_PORT = 9955
CLUSTER_WORKER_PORT = 9956
cluster_cumulative_rewards = {}
lock = Lock()

initial_z_t = None


def rollout(rollout_arg_tuple):
    global initial_z_t
    try:
        generation, mutation_idx, trial, args, vision, model, gpu, W_c, b_c, max_timesteps, with_frames = rollout_arg_tuple

        if initial_z_t is None:
            ope_dir = Path(__file__).parent.resolve()
            random_rollouts_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name,
                                               'random_rollouts')
            if args.in_dream:
                log(ID, "Loading random rollouts for initial frames for dream training")
                initial_z_t = ModelDataset(dir=random_rollouts_dir,
                                           n_rollouts=args.n_rollouts,
                                           load_batch_size=args.initial_z_size,
                                           verbose=False)

        # The same starting seed gets passed in multiprocessing, need to reset it for each process:
        np.random.seed()

        if not with_frames:
            log(ID, ">>> Starting generation #" + str(generation) + ", mutation #" + str(
                mutation_idx + 1) + ", trial #" + str(trial + 1))
        else:
            frames_array = []
        start_time = time.time()

        model.reset_state()

        if args.in_dream:
            z_t, _, _, _, _ = initial_z_t[np.random.randint(len(initial_z_t))]
            z_t = z_t[0]
            if gpu is not None:
                z_t = cuda.to_gpu(z_t)
            if with_frames:
                observation = vision.decode(z_t).data
                if gpu is not None:
                    observation = cp.asnumpy(observation)
                observation = post_process_image_tensor(observation)[0]
            else:
                # free up precious GPU memory:
                if gpu is not None:
                    vision.to_cpu()
                vision = None
            if args.initial_z_noise > 0.:
                if gpu is not None:
                    z_t += cp.random.normal(0., args.initial_z_noise, z_t.shape).astype(cp.float32)
                else:
                    z_t += np.random.normal(0., args.initial_z_noise, z_t.shape).astype(np.float32)
        else:
            if args.game in DOOM_GAMES:
                env = ViZDoomWrapper(args.game)
            else:
                env = gym.make(args.game, render_mode=args.render_mode)
            observation = reset_with_domain(env, args.domain)
        if with_frames:
            frames_array.append(observation)

        if gpu is not None:
            h_t = cp.zeros(args.hidden_dim).astype(cp.float32)
            c_t = cp.zeros(args.hidden_dim).astype(cp.float32)
        else:
            h_t = np.zeros(args.hidden_dim).astype(np.float32)
            c_t = np.zeros(args.hidden_dim).astype(np.float32)

        done = False
        cumulative_reward = 0
        t = 0
        while not done:
            if not args.in_dream:
                observation = cv2.resize(observation, (args.frame_resize, args.frame_resize))
                observation = pre_process_image_tensor(np.expand_dims(observation, 0))

                if gpu is not None:
                    observation = cuda.to_gpu(observation)
                z_t = vision.encode(observation, return_z=True).data[0]

            a_t = action(args, W_c, b_c, z_t, h_t, c_t, gpu)

            if args.in_dream:
                if args.predict_done:
                    z_t, done = model(z_t, a_t, temperature=args.temperature, cpu=args.gpu is None)
                    done = done.data[0]
                else:
                    z_t = model(z_t, a_t, temperature=args.temperature, cpu=args.gpu is None)
                    done = 0.0
                if with_frames:
                    # observation = post_process_image_tensor(vision.to_gpu().decode(z_t).data)[0]
                    z_t.to_cpu()
                    observation = post_process_image_tensor(vision.decode(z_t).data)[0]
                reward = 1
                if done >= args.done_threshold:
                    done = True
                else:
                    done = False
            else:
                observation, reward, done, _, _ = env.step(a_t if gpu is None else cp.asnumpy(a_t))
                if args.render:
                    env.render()
                model(z_t, a_t, temperature=args.temperature, cpu=False)
            if with_frames:
                frames_array.append(observation)

            cumulative_reward += reward

            h_t = model.get_h().data[0]
            c_t = model.get_c().data[0]

            t += 1
            if max_timesteps is not None and t == max_timesteps:
                break
            elif args.in_dream and t == args.dream_max_len:
                log(ID,
                    ">>> generation #{}, mutation #{}, trial #{}: maximum length of {} timesteps reached in dream!"
                    .format(generation, str(mutation_idx + 1), str(trial + 1), t))
                break

        if not args.in_dream:
            env.close()

        if not with_frames:
            log(ID,
                ">>> Finished generation #{}, mutation #{}, trial #{} in {} timesteps in {:.2f}s with cumulative reward {:.2f}"
                .format(generation, str(mutation_idx + 1), str(trial + 1), t, (time.time() - start_time),
                        cumulative_reward))
            return cumulative_reward
        else:
            frames_array = np.asarray(frames_array)
            if args.game in DOOM_GAMES and not args.in_dream:
                frames_array = post_process_image_tensor(frames_array)
            return cumulative_reward, np.asarray(frames_array)
    except Exception:
        print(traceback.format_exc())
        return 0.


def rollout_worker(worker_arg_tuple):
    generation, mutation_idx, args, vision, model, mutation, max_timesteps, in_parallel = worker_arg_tuple
    W_c, b_c = transform_to_weights(args, mutation)

    log(ID, ">> Starting generation #" + str(generation) + ", mutation #" + str(mutation_idx + 1))
    start_time = time.time()

    rollout_arg_tuples = []
    cumulative_rewards = []
    for trial in range(args.trials):
        this_vision = vision.copy()
        this_model = model.copy()
        gpu = None
        if isinstance(args.gpus, (list,)):
            gpu = args.gpus[mutation_idx % len(args.gpus)]
        elif args.gpu >= 0:
            gpu = args.gpu
        if gpu is not None:
            # log(ID,"Assigning GPU "+str(gpu))
            cp.cuda.Device(gpu).use()
            this_vision.to_gpu()
            this_model.to_gpu()
            W_c = cuda.to_gpu(W_c)
            if b_c is not None:
                b_c = cuda.to_gpu(b_c)
        if in_parallel:
            rollout_arg_tuples.append(
                (generation, mutation_idx, trial, args, this_vision, this_model, gpu, W_c, b_c, max_timesteps, False))
        else:
            cumulative_reward = rollout(
                (generation, mutation_idx, trial, args, this_vision, this_model, gpu, W_c, b_c, max_timesteps, False))
            cumulative_rewards.append(cumulative_reward)
    if in_parallel:
        pool = Pool(args.trials)
        cumulative_rewards = pool.map(rollout, rollout_arg_tuples)
        pool.close()
        pool.join()

    avg_cumulative_reward = np.mean(cumulative_rewards)

    log(ID, ">> Finished generation #{}, mutation #{}, in {:.2f}s with averge cumulative reward {:.2f} over {} trials"
        .format(generation, (mutation_idx + 1), (time.time() - start_time), avg_cumulative_reward, args.trials))

    if args.with_wandb:
        wandb.log({'reward': avg_cumulative_reward})

    return avg_cumulative_reward


class WorkerServer(object):
    def __init__(self, port, args, vision, model):
        self.args = args
        self.vision = vision
        self.model = model
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.listen()

    def listen(self):
        self.sock.listen(10)
        while True:
            client, address = self.sock.accept()
            client.settimeout(10)
            Thread(target=self.listenToClient, args=(client, address)).start()

    def listenToClient(self, client, address):
        data = b''
        while True:
            input = client.recv(1024)
            data += input
            if input.endswith(b"\r\n"):
                data = data.strip()
                break
            if not input: break

        npz = np.load(BytesIO(data))
        chunked_mutations = npz['chunked_mutations']
        indices = npz['indices']
        generation = npz['generation']
        max_timesteps = npz['max_timesteps']
        npz.close()
        client.send(b"OK")
        client.close()

        log(ID, "> Received " + str(len(chunked_mutations)) + " mutations from dispatcher")
        length = len(chunked_mutations)
        cores = cpu_count()
        if cores < self.args.trials:
            splits = length
        else:
            splits = math.ceil((length * self.args.trials) / cores)
        chunked_mutations = np.array_split(chunked_mutations, splits)
        indices = np.array_split(indices, splits)
        cumulative_rewards = {}
        for i, this_chunked_mutations in enumerate(chunked_mutations):
            this_indices = indices[i]
            worker_arg_tuples = []
            for i, mutation in enumerate(this_chunked_mutations):
                worker_arg_tuples.append(
                    (generation, this_indices[i], self.args, self.vision, self.model, mutation, max_timesteps, True))
            pool = ThreadPool(len(this_chunked_mutations))
            this_cumulative_rewards = pool.map(rollout_worker, worker_arg_tuples)
            for i, index in enumerate(this_indices):
                cumulative_rewards[index] = this_cumulative_rewards[i]

        log(ID, "> Sending results back to dispatcher: " + str(cumulative_rewards))

        succeeded = False
        for retries in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                sock.connect((CLUSTER_DISPATCHER, CLUSTER_DISPATCHER_PORT))
                sock.sendall(str(cumulative_rewards).encode())
                sock.sendall(b"\r\n")
                data = sock.recv(1024).decode("utf-8")
                sock.close()
                if data == "OK":
                    succeeded = True
                    break
            except Exception as e:
                log(ID, e)
            log(ID, "Unable to send results back to dispatcher. Retrying after sleeping for 30s")
            time.sleep(30)
        if not succeeded:
            log(ID, "Unable to send results back to dispatcher!")


class DispatcherServer(object):
    def __init__(self, port, args, cluster_event):
        self.args = args
        self.cluster_event = cluster_event
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', port))
        self.listen()

    def listen(self):
        try:
            count = 10 * len(CLUSTER_WORKERS)
            self.sock.listen(count)
            while True:
                client, address = self.sock.accept()
                client.settimeout(10)
                Thread(target=self.listenToClient, args=(client, address)).start()
        except Exception as e:
            print(e)

    def listenToClient(self, client, address):
        global cluster_cumulative_rewards
        data = b''
        while True:
            input = client.recv(1024)
            data += input
            if input.endswith(b"\r\n"):
                data = data.strip()
                break
            if not input: break

        cumulative_rewards = ast.literal_eval(data.decode("utf-8"))
        client.send(b"OK")
        client.close()
        log(ID, "> DispatcherServer received results: " + str(cumulative_rewards))
        with lock:
            for index in cumulative_rewards:
                cluster_cumulative_rewards[index] = cumulative_rewards[index]
        if len(cluster_cumulative_rewards) == self.args.lambda_:
            log(ID, "> All results received. Waking up CMA-ES loop")
            self.cluster_event.set()


def main():
    parser = argparse.ArgumentParser(description='World Models ' + ID)
    parser.add_argument('--data_dir', '-d', default="data/wm", help='The base data/output directory')
    parser.add_argument('--game', default='CarRacing-v2',
                        help='Game to use')  # https://www.gymlibrary.dev/environments/box2d/car_racing/
    parser.add_argument('--previous_domain', default=None, type=str, choices=['default', 'B1', 'B2', 'B3'],
                        help='Predefined colors of the previously trained Car Racing environment')
    parser.add_argument('--domain', default='default', type=str, choices=['default', 'B1', 'B2', 'B3'],
                        help='Apply predefined colors for the current Car Racing environment')
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--n_rollouts', default=1000, type=int, help='Number of rollouts to sample for training')
    parser.add_argument('--model', '-m', default='', help='Initialize the model from given file')
    parser.add_argument('--no_resume', action='store_true', help='Don''t auto resume from the latest snapshot')
    parser.add_argument('--resume_from', '-r', default='', help='Resume the optimization from a specific snapshot')
    parser.add_argument('--hidden_dim', default=256, type=int, help='LSTM hidden units')
    parser.add_argument('--z_dim', '-z', default=32, type=int, help='dimension of encoded vector')
    parser.add_argument('--mixtures', default=5, type=int, help='number of gaussian mixtures for MDN')
    parser.add_argument('--lambda_', "-l", default=7, type=int, help='Population size for CMA-ES')
    parser.add_argument('--mu', default=0.5, type=float, help='Keep this percent of fittest mutations for CMA-ES')
    parser.add_argument('--trials', default=3, type=int,
                        help='The number of trials per mutation for CMA-ES, to average fitness score over')
    parser.add_argument('--target_cumulative_reward', default=900, type=int, help='Target cumulative reward')
    parser.add_argument('--frame_resize', default=64, type=int, help='h x w resize of each observation frame')
    parser.add_argument('--temperature', '-t', default=1.0, type=float, help='Temperature (tau) for MDN-RNN (model)')
    parser.add_argument('--snapshot_interval', '-s', default=5, type=int,
                        help='snapshot every x generations of evolution')
    parser.add_argument('--cluster_mode', action='store_true',
                        help='If in a distributed cpu cluster. Set CLUSTER_ variables accordingly.')
    parser.add_argument('--test', action='store_true',
                        help='Generate a rollout gif only (must have access to saved snapshot or model)')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gpus', default="", help='A list of gpus to use, i.e. "0,1,2,3"')
    parser.add_argument('--curriculum', default="", help='initial,step e.g. 50,5 starts at 50 steps and adds 5 steps')
    parser.add_argument('--predict_done', action='store_true', help='Whether MDN-RNN should also predict done state')
    parser.add_argument('--done_threshold', default=0.5, type=float, help='What done probability really means done')
    parser.add_argument('--weights_type', default=1, type=int,
                        help="1=action_dim*(z_dim+hidden_dim), 2=z_dim+2*hidden_dim")
    parser.add_argument('--render_mode', default=None, type=str, choices=['rgb_array', 'human'], help='Render mode')
    parser.add_argument('--in_dream', action='store_true', help='Whether to train in dream, or real environment')
    parser.add_argument('--dream_max_len', default=2100, type=int, help="Maximum timesteps for dream to avoid runaway")
    parser.add_argument('--gif_length', default=1000, type=int, help="Number of timestep to record for the final gif")
    parser.add_argument('--cores', default=0, type=int,
                        help='# CPU cores for main CMA-ES loop in non-cluster_mode. 0=all cores')
    parser.add_argument('--initial_z_size', default=10000, type=int,
                        help="How many real initial frames to load for dream training")
    parser.add_argument('--initial_z_noise', default=0., type=float,
                        help="Gaussian noise std for initial z for dream training")
    parser.add_argument('--cluster_max_wait', default=5400, type=int,
                        help="Move on after this many seconds of no response from worker(s)")

    # WandB
    parser.add_argument('--with_wandb', default=False, action='store_true', help='Enables Weights and Biases')
    parser.add_argument('--wandb_entity', default=None, type=str, help='WandB username (entity).')
    parser.add_argument('--wandb_project', default='COOM', type=str, help='WandB "Project"')
    parser.add_argument('--wandb_group', default='controller', type=str, help='WandB "Group"')
    parser.add_argument('--wandb_job_type', default='train', type=str, help='WandB job type')
    parser.add_argument('--wandb_tags', default=[], type=str, nargs='*', help='Tags can help finding experiments')
    parser.add_argument('--wandb_key', default=None, type=str, help='API key for authorizing WandB')
    parser.add_argument('--wandb_dir', default=None, type=str, help='the place to save WandB files')
    parser.add_argument('--wandb_experiment', default='', type=str, help='Identifier to specify the experiment')

    args = parser.parse_args()
    log(ID, "args =\n " + str(vars(args)).replace(",", ",\n "))

    hostname = socket.gethostname().split(".")[0]
    if args.gpus:
        args.gpus = [int(item) for item in args.gpus.split(',')]
    if args.curriculum:
        curriculum_start = int(args.curriculum.split(',')[0])
        curriculum_step = int(args.curriculum.split(',')[1])

    ope_dir = Path(__file__).parent.resolve()
    controller_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, ID)
    output_dir = os.path.join(controller_dir, args.domain)
    mkdir(output_dir)
    model_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, 'model', args.domain)
    vision_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, 'vision', args.domain)
    random_rollouts_dir = os.path.join(ope_dir, args.data_dir, args.game, args.experiment_name, 'random_rollouts',
                                       args.domain)

    # WandB
    args.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_wandb(ID, args)

    model = MDN_RNN(args.hidden_dim, args.z_dim, args.mixtures, args.predict_done)
    chainer.serializers.load_npz(os.path.join(model_dir, "model.model"), model)
    vision = CVAE(args.z_dim)
    chainer.serializers.load_npz(os.path.join(vision_dir, "vision.model"), vision)

    global initial_z_t
    if args.in_dream:
        log(ID, "Loading random rollouts for initial frames for dream training")
        initial_z_t = ModelDataset(dir=random_rollouts_dir,
                                   n_rollouts=args.n_rollouts,
                                   load_batch_size=args.initial_z_size,
                                   verbose=False)

    env = ViZDoomWrapper(args.game) if args.game in DOOM_GAMES else gym.make(args.game)
    action_dim = len(env.action_space.low)
    args.action_dim = action_dim
    env = None

    auto_resume_file = None
    if not args.cluster_mode or (args.cluster_mode and hostname == CLUSTER_DISPATCHER):
        max_iter = 0
        files = os.listdir(output_dir)
        for file in files:
            if re.match(r'^snapshot_iter_', file):
                iter = int(re.search(r'\d+', file).group())
                if (iter > max_iter):
                    max_iter = iter
        if max_iter > 0:
            auto_resume_file = os.path.join(output_dir, "snapshot_iter_{}.npz".format(max_iter))

    resume = None
    if args.previous_domain:
        args.model = os.path.join(controller_dir, args.previous_domain, ID + ".model")
        log(ID, "Loading saved model from previous domain: " + args.previous_domain)
        resume = args.model
    elif args.model:
        if args.model == 'default':
            args.model = os.path.join(output_dir, ID + ".model")
        log(ID, "Loading saved model from: " + args.model)
        resume = args.model
    elif args.resume_from:
        log(ID, "Resuming manually from snapshot: " + args.resume_from)
        resume = args.resume_from
    elif not args.no_resume and auto_resume_file is not None:
        log(ID, "Auto resuming from last snapshot: " + auto_resume_file)
        resume = auto_resume_file

    if resume is not None:
        npz = np.load(resume)
        pc = npz['pc']
        ps = npz['ps']
        B = npz['B']
        D = npz['D']
        C = npz['C']
        invsqrtC = npz['invsqrtC']
        eigeneval = npz['eigeneval']
        xmean = npz['xmean']
        sigma = npz['sigma']
        counteval = npz['counteval']
        generation = npz['generation'] + 1
        cumulative_rewards_over_generations = npz['cumulative_rewards_over_generations']
        if args.curriculum:
            if 'max_timesteps' in npz and npz['max_timesteps'] is not None:
                max_timesteps = npz['max_timesteps']
            else:
                max_timesteps = curriculum_start
            last_highest_avg_cumulative_reward = max(cumulative_rewards_over_generations.mean(axis=1))
        else:
            max_timesteps = None
        npz.close()

    log(ID, "Starting")

    if args.cluster_mode and hostname != CLUSTER_DISPATCHER and not args.test:
        log(ID, "Starting cluster worker")
        WorkerServer(CLUSTER_WORKER_PORT, args, vision, model)
    elif not args.test:
        if args.cluster_mode:
            global cluster_cumulative_rewards
            cluster_event = Event()

            log(ID, "Starting cluster dispatcher")
            dispatcher_thread = Thread(target=DispatcherServer, args=(CLUSTER_DISPATCHER_PORT, args, cluster_event))
            dispatcher_thread.start()

            # Make the dispatcher a worker too
            log(ID, "Starting cluster worker")
            worker_thread = Thread(target=WorkerServer, args=(CLUSTER_WORKER_PORT, args, vision, model))
            worker_thread.start()

        if args.weights_type == 1:
            N = action_dim * (args.z_dim + args.hidden_dim) + action_dim
        elif args.weights_type == 2:
            N = args.z_dim + 2 * args.hidden_dim

        stopeval = 1e3 * N**2
        stopfitness = args.target_cumulative_reward

        lambda_ = args.lambda_  # 4+int(3*np.log(N))
        mu = int(lambda_ * args.mu)  # //2
        weights = np.log(mu + 1 / 2) - np.log(np.asarray(range(1, mu + 1))).astype(np.float32)
        weights = weights / np.sum(weights)
        mueff = (np.sum(weights)**2) / np.sum(weights**2)

        cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N)
        cs = (mueff + 2) / (N + mueff + 5)
        c1 = 2 / ((N + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2)**2 + mueff))
        damps = 1 + 2 * max(0, ((mueff - 1) / (N + 1))**0.5 - 1) + cs
        chiN = N**0.5 * (1 - 1 / (4 * N) + 1 / (21 * N**2))

        if resume is None:
            pc = np.zeros(N).astype(np.float32)
            ps = np.zeros(N).astype(np.float32)
            B = np.eye(N, N).astype(np.float32)
            D = np.ones(N).astype(np.float32)
            C = B * np.diag(D**2) * B.T
            invsqrtC = B * np.diag(D**-1) * B.T
            eigeneval = 0
            xmean = np.random.randn(N).astype(np.float32)
            sigma = 0.3
            counteval = 0
            generation = 1
            cumulative_rewards_over_generations = None
            if args.curriculum:
                max_timesteps = curriculum_start
                last_highest_avg_cumulative_reward = None
            else:
                max_timesteps = None

        solution_found = False
        while counteval < stopeval:
            log(ID, "> Starting evolution generation #" + str(generation))

            arfitness = np.zeros(lambda_).astype(np.float32)
            arx = np.zeros((lambda_, N)).astype(np.float32)
            for k in range(lambda_):
                arx[k] = xmean + sigma * B.dot(D * np.random.randn(N).astype(np.float32))
                counteval += 1

            if not args.cluster_mode:
                if args.cores == 0:
                    cores = cpu_count()
                else:
                    cores = args.cores
                pool = Pool(cores)
                worker_arg_tuples = []
                for k in range(lambda_):
                    worker_arg_tuples.append((generation, k, args, vision, model, arx[k], max_timesteps, False))
                cumulative_rewards = pool.map(rollout_worker, worker_arg_tuples)
                pool.close()
                pool.join()
                for k, cumulative_reward in enumerate(cumulative_rewards):
                    arfitness[k] = cumulative_reward
            else:
                arx_splits = np.array_split(arx, len(CLUSTER_WORKERS))
                indices = np.array_split(np.arange(lambda_), len(CLUSTER_WORKERS))
                cluster_cumulative_rewards = {}
                for i, chunked_mutations in enumerate(arx_splits):
                    log(ID, "> Dispatching " + str(len(chunked_mutations)) + " mutations to " + CLUSTER_WORKERS[i])
                    compressed_array = BytesIO()
                    np.savez_compressed(compressed_array,
                                        chunked_mutations=chunked_mutations,
                                        indices=indices[i],
                                        generation=generation,
                                        max_timesteps=max_timesteps)
                    compressed_array.seek(0)
                    out = compressed_array.read()

                    succeeded = False
                    for retries in range(3):
                        try:
                            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            sock.settimeout(10)
                            sock.connect((CLUSTER_WORKERS[i], CLUSTER_WORKER_PORT))
                            sock.sendall(out)
                            sock.sendall(b"\r\n")
                            data = sock.recv(1024).decode("utf-8")
                            sock.close()
                            if data == "OK":
                                succeeded = True
                                break
                        except Exception as e:
                            log(ID, e)
                        log(ID, "Unable to dispatch mutations to " + CLUSTER_WORKERS[
                            i] + ". Retrying after sleeping for 30s")
                        time.sleep(30)
                    if not succeeded:
                        log(ID, "Unable to dispatch mutations to " + CLUSTER_WORKERS[i] + "!")
                log(ID, "> Dispatched all mutations to cluster. Waiting for results.")
                cluster_event.clear()
                cluster_event.wait(args.cluster_max_wait)  # Cut our losses if some results never get returned
                for k in range(lambda_):
                    if k in cluster_cumulative_rewards:
                        arfitness[k] = cluster_cumulative_rewards[k]
                    else:
                        arfitness[k] = 0.

            if cumulative_rewards_over_generations is None:
                cumulative_rewards_over_generations = np.expand_dims(arfitness, 0)
            else:
                cumulative_rewards_over_generations = np.concatenate(
                    (cumulative_rewards_over_generations, np.expand_dims(arfitness, 0)),
                    axis=0)

            arindex = np.argsort(-arfitness)
            # arfitness = arfitness[arindex]

            xold = xmean
            xmean = weights.dot(arx[arindex[0:mu]])

            avg_cumulative_reward = np.mean(arfitness)

            log(ID, "> Finished evolution generation #{}, average cumulative reward = {:.2f}"
                .format(generation, avg_cumulative_reward))

            if generation > 1 and args.curriculum:
                if last_highest_avg_cumulative_reward is None:
                    last_highest_avg_cumulative_reward = np.mean(cumulative_rewards_over_generations[-2])
                log(ID, "> Highest average cumulative reward from previous generations = {:.2f}".format(
                    last_highest_avg_cumulative_reward))
                if avg_cumulative_reward > (
                        last_highest_avg_cumulative_reward * 0.99):  # Let is pass if within 1% of the old average
                    max_timesteps += curriculum_step
                    log(ID, "> Average cumulative reward increased. Increasing max timesteps to " + str(max_timesteps))
                    last_highest_avg_cumulative_reward = None
                else:
                    log(ID,
                        "> Average cumulative reward did not increase. Keeping max timesteps at " + str(max_timesteps))

            # Average over the whole population, but breaking here means we use only the
            # top x% of the mutations as the calculation for the final mean
            if avg_cumulative_reward >= stopfitness:
                solution_found = True
                break

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC.dot((xmean - xold) / sigma)
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * counteval / lambda_)) / chiN < 1.4 + 2 / (N + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * ((xmean - xold) / sigma)
            artmp = (1 / sigma) * (arx[arindex[0:mu]] - xold)
            C = (1 - c1 - cmu) * C + c1 * (pc.dot(pc.T) + (1 - hsig) * cc * (2 - cc) * C) + cmu * artmp.T.dot(
                np.diag(weights)).dot(artmp)
            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))

            if counteval - eigeneval > lambda_ / (c1 + cmu) / N / 10:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eig(C)
                D = np.sqrt(D)
                invsqrtC = B.dot(np.diag(D**-1).dot(B.T))

            if generation % args.snapshot_interval == 0:
                snapshot_file = os.path.join(output_dir, "snapshot_iter_" + str(generation) + ".npz")
                log(ID, "> Saving snapshot to " + str(snapshot_file))
                np.savez_compressed(snapshot_file,
                                    pc=pc,
                                    ps=ps,
                                    B=B,
                                    D=D,
                                    C=C,
                                    invsqrtC=invsqrtC,
                                    eigeneval=eigeneval,
                                    xmean=xmean,
                                    sigma=sigma,
                                    counteval=counteval,
                                    generation=generation,
                                    cumulative_rewards_over_generations=cumulative_rewards_over_generations,
                                    max_timesteps=max_timesteps)

            generation += 1

        if solution_found:
            log(ID, "Evolution Complete!")
            log(ID, "Solution found at generation #" + str(generation) + ", with average cumulative reward = " +
                str(avg_cumulative_reward) + " over " + str(args.lambda_ * args.trials) + " rollouts")
        else:
            log(ID, "Solution not found")

        controller_model_file = os.path.join(output_dir, ID + ".model")
        if os.path.exists(controller_model_file):
            os.remove(controller_model_file)
        log(ID, "Saving model to: " + controller_model_file)
        np.savez_compressed(controller_model_file,
                            pc=pc,
                            ps=ps,
                            B=B,
                            D=D,
                            C=C,
                            invsqrtC=invsqrtC,
                            eigeneval=eigeneval,
                            xmean=xmean,
                            sigma=sigma,
                            counteval=counteval,
                            generation=generation,
                            cumulative_rewards_over_generations=cumulative_rewards_over_generations,
                            max_timesteps=max_timesteps)
        os.rename(os.path.join(output_dir, ID + ".model.npz"), controller_model_file)

    # xmean = np.random.randn(action_dim * (args.z_dim + args.hidden_dim) + action_dim).astype(np.float32)
    # xmean = np.random.randn(args.z_dim + 2 * args.hidden_dim).astype(np.float32)
    parameters = xmean

    if args.in_dream:
        log(ID, "Generating a rollout gif with the controller model in a dream")
        W_c, b_c = transform_to_weights(args, parameters)
        cumulative_reward, frames = rollout(
            (0, 0, 0, args, vision.to_cpu(), model.to_cpu(), None, W_c, b_c, None, True))
        imageio.mimsave(os.path.join(output_dir, 'dream_rollout.gif'), frames, fps=20)
        log(ID, "Final cumulative reward in dream: " + str(cumulative_reward))
        args.in_dream = False

    log(ID, "Generating a rollout gif with the controller model in the environment")
    W_c, b_c = transform_to_weights(args, parameters)
    cumulative_reward, frames = rollout(
        (0, 0, 0, args, vision.to_cpu(), model.to_cpu(), None, W_c, b_c, args.gif_length, True))
    imageio.mimsave(os.path.join(output_dir, 'env_rollout.gif'), frames, fps=20)
    log(ID, "Final cumulative reward in environment: " + str(cumulative_reward))

    log(ID, "Done")


if __name__ == '__main__':
    main()
