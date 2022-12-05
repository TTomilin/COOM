import argparse
import math
import os
import re
from itertools import count

import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
import cupy as cp
import gym
import imageio
import numba
import numpy as np
import scipy.stats
from chainer import Chain
from chainer import optimizers
from chainer import training
from chainer.backends import cuda
from chainer.training import extensions

from lib.data import ModelDataset
from lib.utils import log, mkdir, save_images_collage, post_process_image_tensor
from vision import CVAE

ID = "model"


class PolicyNet(Chain):

    def __init__(self, args):
        super(PolicyNet, self).__init__()
        with self.init_scope():
            self.args = args
            self.W_c = L.Linear(None, args.action_dim)

    def forward(self, args, z_t, h_t, c_t):
        if args.weights_type == 1:
            input = F.concat((z_t, h_t), axis=0).data
            input = F.reshape(input, (1, input.shape[0]))
            action = F.tanh(self.W_c(input)).data
        elif args.weights_type == 2:
            input = F.concat((z_t, h_t, c_t), axis=0).data
            dot = self.W_c(input)
            if args.gpu >= 0:
                dot = cp.asarray(dot)
            else:
                dot = np.asarray(dot)
            output = F.tanh(dot)
            if output == 1.:
                output = 0.999
            action_dim = args.action_dim + 1
            action_range = 2 / action_dim
            action = [0. for i in range(action_dim)]
            start = -1.
            for i in range(action_dim):
                if start <= output and output <= (start + action_range):
                    action[i] = 1.
                    break
                start += action_range
            mid = action_dim // 2  # reserve action[mid] for no action
            action = action[0:mid] + action[mid + 1:action_dim]
        if args.gpu >= 0:
            action = cp.asarray(action).astype(cp.float32)
        else:
            action = np.asarray(action).astype(np.float32)
        return action


def train_lgc(args, model):
    """
    Train a stochastic (Gaussian) policy that acts on [z_t,h_t] in a virtual world dictated by model
    Use policy gradient.
    :param args:
    :param vision:
    :param model:
    :return: coefficients of linear controller, W_c and b_c in W_c [z_t,h_t] + b_c
    """
    episode_durations = []

    random_rollouts_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'random_rollouts')
    initial_z_t = ModelDataset(dir=random_rollouts_dir,
                               load_batch_size=args.initial_z_size,
                               verbose=False)

    num_episode = 10
    batch_size = 5
    gamma = 0.99

    policy_net = PolicyNet(args)
    optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(policy_net)

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    for e in range(num_episode):

        # grab initial state tuple (z_t, h_t, c_t) from historical random rollouts
        z_t, _, _, _, _ = initial_z_t[np.random.randint(len(initial_z_t))]
        z_t = z_t[0]
        if args.gpu >= 0:
            z_t = cuda.to_gpu(z_t)
        if args.initial_z_noise > 0.:
            if args.gpu >= 0:
                z_t += cp.random.normal(0., args.initial_z_noise, z_t.shape).astype(cp.float32)
            else:
                z_t += np.random.normal(0., args.initial_z_noise, z_t.shape).astype(np.float32)
        if args.gpu >= 0:
            h_t = cp.zeros(args.hidden_dim).astype(cp.float32)
            c_t = cp.zeros(args.hidden_dim).astype(cp.float32)
        else:
            h_t = np.zeros(args.hidden_dim).astype(np.float32)
            c_t = np.zeros(args.hidden_dim).astype(np.float32)

        for t in count():

            mean_a_t = policy_net(args, z_t, h_t, c_t)
            action_policy_std = 0.1
            cov = action_policy_std * np.identity(args.action_dim)
            stochastic_policy = D.MultivariateNormal(loc=mean_a_t.astype(np.float32), scale_tril=cov.astype(np.float32))
            a_t = stochastic_policy.sample()

            z_t, done = model(z_t, a_t, temperature=args.temperature)
            done = done.data[0]
            reward = 1.0
            if done >= args.done_threshold or t >= args.max_episode_length:
                done = True
            else:
                done = False

            h_t = model.get_h().data[0]
            c_t = model.get_c().data[0]

            state_pool.append((z_t, h_t, c_t))
            action_pool.append(a_t)
            reward_pool.append(reward)

            steps += 1

            if done:
                episode_durations.append(t + 1)
                break

        # Update policy
        if e > 0 and e % batch_size == 0:

            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * gamma + reward_pool[i]
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            policy_net.cleargrads()

            for i in range(steps):
                z_t,h_t,c_t = state_pool[i]
                action = action_pool[i]
                reward = reward_pool[i]

                mean_a_t = policy_net(args, z_t, h_t, c_t)
                action_policy_std = 0.1
                cov = action_policy_std * np.identity(args.action_dim)
                stochastic_policy = D.MultivariateNormal(loc=mean_a_t.astype(np.float32),
                                                         scale_tril=cov.astype(np.float32))
                loss = -stochastic_policy.log_prob(action) * reward  # Negtive score function x reward
                loss.backward()

            optimizer.update()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0

    return policy_net


def ope_LGC(args, model, policy_net):
    """
    Off-policy policy evaluation of a linear Gaussian controller
    acting as N(W_c * [z_t,h_t] + b_c, 0.1 * identity matrix(args.action_dim))
    This is done using "historical" data in random_rollouts directory
    :param args:
    :param model:
    :param W_c:
    :param b_c:
    :return:
    """
    random_rollouts_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'random_rollouts')
    train = ModelDataset(dir=random_rollouts_dir, load_batch_size=args.load_batch_size, verbose=False)

    ope = 0

    for i in range(len(train)):
        h_t = np.zeros(args.hidden_dim).astype(np.float32)
        c_t = np.zeros(args.hidden_dim).astype(np.float32)
        t = 0

        rollout_z_t, rollout_z_t_plus_1, rollout_action, rollout_reward, rollout_done = train[i]  # Pick a real rollout
        done = rollout_done[0]
        weight_prod = 1

        while not done:

            z_t = rollout_z_t[t]

            eval_policy_mean = policy_net(args, z_t, h_t, c_t)
            eval_policy_mean = np.transpose(eval_policy_mean).reshape(args.action_dim)

            # TODO: yikes this shouldn't be hardcoded...
            action_policy_std = 0.1
            #
            weight_prod *= scipy.stats.multivariate_normal.pdf(rollout_action[t], eval_policy_mean,
                                                               action_policy_std * np.identity(args.action_dim))
            ope += weight_prod * rollout_reward[t]

            model(z_t, rollout_action[t], temperature=args.temperature)
            h_t = model.get_h().data[0]
            c_t = model.get_c().data[0]

            t += 1
            done = rollout_done[t]

    return ope

@numba.jit(nopython=True)
def optimized_sampling(output_dim, temperature, coef, mu, ln_var):
    mus = np.zeros(output_dim)
    ln_vars = np.zeros(output_dim)
    for i in range(output_dim):
        cumulative_probability = 0.
        r = np.random.uniform(0., 1.)
        index = len(coef)-1
        for j, probability in enumerate(coef[i]):
            cumulative_probability = cumulative_probability + probability
            if r <= cumulative_probability:
                index = j
                break
        for j, this_mu in enumerate(mu[i]):
            if j == index:
                mus[i] = this_mu
                break
        for j, this_ln_var in enumerate(ln_var[i]):
            if j == index:
                ln_vars[i] = this_ln_var
                break
    z_t_plus_1 = mus + np.exp(ln_vars) * np.random.randn(output_dim) * np.sqrt(temperature)
    return z_t_plus_1


class MDN_RNN(chainer.Chain):
    def __init__(self, hidden_dim=256, output_dim=32, k=5, predict_done=False):
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self._cpu = True
        self.predict_done = predict_done
        init_dict = {
            "rnn_layer": L.LSTM(None, hidden_dim),
            "coef_layer": L.Linear(None, k * output_dim),
            "mu_layer": L.Linear(None, k * output_dim),
            "ln_var_layer": L.Linear(None, k * output_dim)
        }
        if predict_done:
            init_dict["done_layer"] = L.Linear(None, 1)
        super(MDN_RNN, self).__init__(**init_dict)

    def __call__(self, z_t, action, temperature=1.0):
        k = self.k
        output_dim = self.output_dim

        if len(z_t.shape) == 1:
            z_t = F.expand_dims(z_t, 0)
        if len(action.shape) == 1:
            action = F.expand_dims(action, 0)

        output = self.fprop(F.concat((z_t, action)))
        if self.predict_done:
            coef, mu, ln_var, done = output
        else:
            coef, mu, ln_var = output

        coef = F.reshape(coef, (-1, k))
        mu = F.reshape(mu, (-1, k))
        ln_var = F.reshape(ln_var, (-1, k))

        coef /= temperature
        coef = F.softmax(coef,axis=1)

        if self._cpu:
            z_t_plus_1 = optimized_sampling(output_dim, temperature, coef.data, mu.data, ln_var.data).astype(np.float32)
        else:
            coef = cp.asnumpy(coef.data)
            mu = cp.asnumpy(mu.data)
            ln_var = cp.asnumpy(ln_var.data)
            z_t_plus_1 = optimized_sampling(output_dim, temperature, coef, mu, ln_var).astype(np.float32)
            z_t_plus_1 = chainer.Variable(cp.asarray(z_t_plus_1))

        if self.predict_done:
            return z_t_plus_1, F.sigmoid(done)
        else:
            return z_t_plus_1

    def fprop(self, input):
        h = self.rnn_layer(input)
        coef = self.coef_layer(h)
        mu = self.mu_layer(h)
        ln_var = self.ln_var_layer(h)

        if self.predict_done:
            done = self.done_layer(h)

        if self.predict_done:
            return coef, mu, ln_var, done
        else:
            return coef, mu, ln_var

    def get_loss_func(self):
        def lf(z_t, z_t_plus_1, action, done_label, reset=True):
            k = self.k
            output_dim = self.output_dim
            if reset:
                self.reset_state()

            output = self.fprop(F.concat((z_t, action)))
            if self.predict_done:
                coef, mu, ln_var, done = output
            else:
                coef, mu, ln_var = output

            coef = F.reshape(coef, (-1, output_dim, k))
            coef = F.softmax(coef, axis=2)
            mu = F.reshape(mu, (-1, output_dim, k))
            ln_var = F.reshape(ln_var, (-1, output_dim, k))

            z_t_plus_1 = F.repeat(z_t_plus_1, k, 1).reshape(-1, output_dim, k)

            normals = F.sum(
                coef * F.exp(-F.gaussian_nll(z_t_plus_1, mu, ln_var, reduce='no'))
                ,axis=2)
            densities = F.sum(normals, axis=1)
            nll = -F.log(densities)

            loss = F.sum(nll)

            if self.predict_done:
                done_loss = F.sigmoid_cross_entropy(done.reshape(-1,1), done_label, reduce="no")
                done_loss *= (1. + done_label.astype("float32")*9.)
                done_loss = F.mean(done_loss)
                loss = loss + done_loss

            return loss
        return lf

    def reset_state(self):
        self.rnn_layer.reset_state()

    def get_h(self):
        return self.rnn_layer.h

    def get_c(self):
        return self.rnn_layer.c


class ImageSampler(chainer.training.Extension):
    def __init__(self, model, vision, args, output_dir, z_t, action):
        self.model = model
        self.vision = vision
        self.args = args
        self.output_dir = output_dir
        self.z_t = z_t
        self.action = action

    def __call__(self, trainer):
        if self.args.gpu >= 0:
            self.model.to_cpu()
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            self.model.reset_state()
            z_t_plus_1s = []
            dones = []
            for i in range(self.z_t.shape[0]):
                output = self.model(self.z_t[i], self.action[i], temperature=self.args.sample_temperature)
                if self.args.predict_done:
                    z_t_plus_1, done = output
                    z_t_plus_1 = z_t_plus_1.data
                    done = done.data
                else:
                    z_t_plus_1 = output.data
                z_t_plus_1s.append(z_t_plus_1)
                if self.args.predict_done:
                    dones.append(done[0])
            z_t_plus_1s = np.asarray(z_t_plus_1s)
            dones = np.asarray(dones).reshape(-1)
            img_t_plus_1 = post_process_image_tensor(self.vision.decode(z_t_plus_1s).data)
            if self.args.predict_done:
                img_t_plus_1[np.where(dones >= 0.5), :, :, :] = 0  # Make all the done's black
            save_images_collage(img_t_plus_1,
                                os.path.join(self.output_dir,
                                             'train_t_plus_1_{}.png'.format(trainer.updater.iteration)),
                                pre_processed=False)
        if self.args.gpu >= 0:
            self.model.to_gpu()


class TBPTTUpdater(training.updaters.StandardUpdater):
    def __init__(self, train_iter, optimizer, loss_func, args, model):
        self.sequence_length = args.sequence_length
        self.args = args
        self.model = model
        # self.device = args.gpu
        # self.device = 0
        super(TBPTTUpdater, self).__init__(train_iter, optimizer,loss_func=loss_func)

    def update_core(self):

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Train linear Gaussian controller policy given the current latent space transition model, self.model
        policy_net = train_lgc(self.args, self.model)

        # Evaluate linear Gaussian controller on historical data
        ope = ope_LGC(self.args, self.model, policy_net)

        batch = train_iter.__next__()
        total_loss = 0
        z_t, z_t_plus_1, action, _, done = self.converter(batch, self.device)
        z_t = chainer.Variable(z_t[0])
        z_t_plus_1 = chainer.Variable(z_t_plus_1[0])
        action = chainer.Variable(action[0])
        done = chainer.Variable(done[0])
        for i in range(math.ceil(z_t.shape[0]/self.sequence_length)):
            start_idx = i*self.sequence_length
            end_idx = (i+1)*self.sequence_length
            loss = self.loss_func(z_t[start_idx:end_idx].data,
                                  z_t_plus_1[start_idx:end_idx].data,
                                  action[start_idx:end_idx].data,
                                  done[start_idx:end_idx].data,
                                  True if i==0 else False)

            # TODO: should the adversarial ope loss be subtracted this many times during the for loop? Should not hardcode ope_scale
            ope_scale = 100
            loss -= ope_scale*ope

            optimizer.target.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
            total_loss += loss

        chainer.report({'loss': total_loss})


def main():
    parser = argparse.ArgumentParser(description='World Models ' + ID)
    parser.add_argument('--data_dir', '-d', default="./data/wm", help='The base data/output directory')
    parser.add_argument('--game', default='CarRacing-v0',
                        help='Game to use')  # https://gym.openai.com/envs/CarRacing-v0/
    parser.add_argument('--experiment_name', default='experiment_1', help='To isolate its files from others')
    parser.add_argument('--load_batch_size', default=100, type=int,
                        help='Load rollouts in batches so as not to run out of memory')
    parser.add_argument('--model', '-m', default='',
                        help='Initialize the model from given file, or "default" for one in data folder')
    parser.add_argument('--no_resume', action='store_true', help='Don''t auto resume from the latest snapshot')
    parser.add_argument('--resume_from', '-r', default='', help='Resume the optimization from a specific snapshot')
    parser.add_argument('--test', action='store_true', help='Generate samples only')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=20, type=int, help='number of epochs to learn')
    parser.add_argument('--snapshot_interval', '-s', default=200, type=int, help='snapshot every x games')
    parser.add_argument('--z_dim', '-z', default=32, type=int, help='dimension of encoded vector')
    parser.add_argument('--hidden_dim', default=256, type=int, help='LSTM hidden units')
    parser.add_argument('--mixtures', default=5, type=int, help='number of gaussian mixtures for MDN')
    parser.add_argument('--no_progress_bar', '-p', action='store_true', help='Display progress bar during training')
    parser.add_argument('--predict_done', action='store_true', help='Whether MDN-RNN should also predict done state')
    parser.add_argument('--sample_temperature', default=1., type=float, help='Temperature for generating samples')
    parser.add_argument('--gradient_clip', default=0., type=float, help='Clip grads L2 norm threshold. 0 = no clip')
    parser.add_argument('--sequence_length', type=int, default=128, help='sequence length for LSTM for TBPTT')
    parser.add_argument('--in_dream', action='store_true', help='Whether to train in dream, or real environment')
    parser.add_argument('--initial_z_noise', default=0., type=float,
                        help="Gaussian noise std for initial z for dream training")
    parser.add_argument('--done_threshold', default=0.5, type=float, help='What done probability really means done')
    parser.add_argument('--temperature', '-t', default=1.0, type=float, help='Temperature (tau) for MDN-RNN (model)')
    parser.add_argument('--dream_max_len', default=2100, type=int, help="Maximum timesteps for dream to avoid runaway")
    parser.add_argument('--max_episode_length', default=100, type=int, help="Maximum timesteps for real env")
    parser.add_argument('--weights_type', default=1, type=int,
                        help="1=action_dim*(z_dim+hidden_dim), 2=z_dim+2*hidden_dim")
    parser.add_argument('--initial_z_size', default=10000, type=int,
                        help="How many real initial frames to load for dream training")

    args = parser.parse_args()
    log(ID, "args =\n " + str(vars(args)).replace(",", ",\n "))


    output_dir = os.path.join(args.data_dir, args.game, args.experiment_name, ID)
    mkdir(output_dir)
    random_rollouts_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'random_rollouts')
    vision_dir = os.path.join(args.data_dir, args.game, args.experiment_name, 'vision')


    log(ID, "Starting")

    max_iter = 0
    auto_resume_file = None
    files = os.listdir(output_dir)
    for file in files:
        if re.match(r'^snapshot_iter_', file):
            iter = int(re.search(r'\d+', file).group())
            if (iter > max_iter):
                max_iter = iter
    if max_iter > 0:
        auto_resume_file = os.path.join(output_dir, "snapshot_iter_{}".format(max_iter))

    model = MDN_RNN(args.hidden_dim, args.z_dim, args.mixtures, args.predict_done)
    vision = CVAE(args.z_dim)
    chainer.serializers.load_npz(os.path.join(vision_dir, "vision.model"), vision)

    if args.model:
        if args.model == 'default':
            args.model = os.path.join(output_dir, ID + ".model")
        log(ID, "Loading saved model from: " + args.model)
        chainer.serializers.load_npz(args.model, model)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    if args.gradient_clip > 0.:
        optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(args.gradient_clip))

    log(ID, "Loading training data")
    train = ModelDataset(dir=random_rollouts_dir, load_batch_size=args.load_batch_size, verbose=False)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=1, shuffle=False)


    env = gym.make(args.game)
    action_dim = len(env.action_space.low)
    args.action_dim = action_dim


    updater = TBPTTUpdater(train_iter, optimizer, model.get_loss_func(), args, model)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=output_dir)
    trainer.extend(extensions.snapshot(), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(trigger=(10 if args.gpu >= 0 else 1, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'elapsed_time']))
    if not args.no_progress_bar:
        trainer.extend(extensions.ProgressBar(update_interval=10 if args.gpu >= 0 else 1))

    sample_size = 256
    rollout_z_t, rollout_z_t_plus_1, rollout_action, _, done = train[0]
    sample_z_t = rollout_z_t[0:sample_size]
    sample_z_t_plus_1 = rollout_z_t_plus_1[0:sample_size]
    sample_action = rollout_action[0:sample_size]
    img_t = vision.decode(sample_z_t).data
    img_t_plus_1 = vision.decode(sample_z_t_plus_1).data
    if args.predict_done:
        done = done.reshape(-1)
        img_t_plus_1[np.where(done[0:sample_size] >= 0.5), :, :, :] = 0 # Make done black
    save_images_collage(img_t, os.path.join(output_dir, 'train_t.png'))
    save_images_collage(img_t_plus_1, os.path.join(output_dir, 'train_t_plus_1.png'))
    image_sampler = ImageSampler(model.copy(), vision, args, output_dir, sample_z_t, sample_action)
    trainer.extend(image_sampler, trigger=(args.snapshot_interval, 'iteration'))

    if args.resume_from:
        log(ID, "Resuming trainer manually from snapshot: " + args.resume_from)
        chainer.serializers.load_npz(args.resume_from, trainer)
    elif not args.no_resume and auto_resume_file is not None:
        log(ID, "Auto resuming trainer from last snapshot: " + auto_resume_file)
        chainer.serializers.load_npz(auto_resume_file, trainer)

    if not args.test:
        log(ID, "Starting training")
        trainer.run()
        log(ID, "Done training")
        log(ID, "Saving model")
        chainer.serializers.save_npz(os.path.join(output_dir, ID + ".model"), model)

    if args.test:
        log(ID, "Saving test samples")
        image_sampler(trainer)

    log(ID, "Generating gif for a rollout generated in dream")
    if args.gpu >= 0:
        model.to_cpu()
    model.reset_state()
    # current_z_t = np.random.randn(64).astype(np.float32)  # Noise as starting frame
    rollout_z_t, rollout_z_t_plus_1, rollout_action, rewards, done = train[np.random.randint(len(train))]  # Pick a random real rollout
    current_z_t = rollout_z_t[0] # Starting frame from the real rollout
    current_z_t += np.random.normal(0, 0.5, current_z_t.shape).astype(np.float32)  # Add some noise to the real rollout starting frame
    all_z_t = [current_z_t]
    # current_action = np.asarray([0., 1.]).astype(np.float32)
    for i in range(rollout_z_t.shape[0]):
        # if i != 0 and i % 200 == 0: current_action = 1 - current_action  # Flip actions every 100 frames
        current_action = np.expand_dims(rollout_action[i], 0)  # follow actions performed in a real rollout
        output = model(current_z_t, current_action, temperature=args.sample_temperature)
        if args.predict_done:
            current_z_t, done = output
            done = done.data
            # print(i, current_action, done)
        else:
            current_z_t = output
        all_z_t.append(current_z_t.data)
        if args.predict_done and done[0] >= 0.5:
            break
    dream_rollout_imgs = vision.decode(np.asarray(all_z_t).astype(np.float32)).data
    dream_rollout_imgs = post_process_image_tensor(dream_rollout_imgs)
    imageio.mimsave(os.path.join(output_dir, 'dream_rollout.gif'), dream_rollout_imgs, fps=20)

    log(ID, "Done")


if __name__ == '__main__':
    main()