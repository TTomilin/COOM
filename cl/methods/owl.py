import numpy as np
import tensorflow as tf
import time

from cl.methods.ewc import EWC_SAC
from cl.utils.run_utils import create_one_hot_vec


class ExpWeights(object):

    def __init__(self,
                 arms=[0, 1],
                 lr=1,
                 window=20,  # we don't use this yet..
                 epsilon=0,  # set this above zero for
                 decay=1,
                 greedy=False):

        self.arms = arms
        self.l = {i: 0 for i in range(len(self.arms))}
        self.p = [0.5] * len(self.arms)
        self.arm = 0
        self.value = self.arms[self.arm]
        self.error_buffer = []
        self.window = window
        self.lr = lr
        self.epsilon = epsilon
        self.decay = decay
        self.greedy = greedy

        self.choices = [self.arm]
        self.data = []

    def sample(self):

        if np.random.uniform() > self.epsilon:
            self.p = [np.exp(x) for x in self.l.values()]
            self.p /= np.sum(self.p)  # normalize to make it a distribution
            try:
                self.arm = np.random.choice(range(0, len(self.p)), p=self.p)
            except ValueError:
                print("loss too large scaling")
                decay = self.lr * 0.1
                self.p = [np.exp(x * decay) for x in self.l.values()]
                self.p /= np.sum(self.p)  # normalize to make it a distribution
                self.arm = np.random.choice(range(0, len(self.p)), p=self.p)
        else:
            self.arm = int(np.random.uniform() * len(self.arms))

        self.value = self.arms[self.arm]
        self.choices.append(self.arm)

        return (self.value)

    def update_dists(self, feedback, norm=1):

        # feedback is a list

        for i in range(len(self.arms)):
            if self.greedy:
                self.l[i] *= self.decay
                self.l[i] += self.lr * feedback[i]
            else:
                self.l[i] *= self.decay
                self.l[i] += self.lr * (feedback[i] / max(np.exp(self.l[i]), 0.0001))


# noinspection PyInterpreter
class OWL_SAC(EWC_SAC):
    """OWL method.

    https://arxiv.org/abs/2106.02940"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def test_agent(self, deterministic: bool, num_episodes: int) -> None:
        mode = "deterministic" if deterministic else "stochastic"
        num_actions = self.test_envs[0].action_space.n
        total_action_counts = {i: 0 for i in range(num_actions)}

        # Bandit
        lr = 0.90
        decay = 0.90
        epsilon = 0.0
        bandit_step = 1
        greedy_bandit = True
        bandit_loss = 'mse'

        # Bandit debug
        n_tasks = len(self.test_envs)
        n_arms = n_tasks
        episode_max_steps = self.env.get_active_env().episode_timeout
        feedback, arm = np.empty((n_tasks, n_arms, num_episodes, episode_max_steps + 1)), \
                        np.empty((n_tasks, num_episodes, episode_max_steps + 1))
        mses = np.empty((n_tasks, n_arms, num_episodes, episode_max_steps + 1))
        feedback[:], arm[:], mses[:] = np.nan, np.nan, 0

        # TB
        bandit_probs, bandit_p = np.empty((n_tasks, n_arms, num_episodes)), np.empty(
            (n_tasks, n_arms, num_episodes, episode_max_steps + 1))
        bandit_probs[:], bandit_p[:] = np.nan, np.nan
        dones, corrects = {i: 0 for i in range(n_tasks)}, {i: [] for i in range(n_tasks)}
        return_per_episode, num_frames_per_episode = \
            np.zeros((n_tasks, num_episodes)), np.zeros((n_tasks, num_episodes))

        for seq_idx, test_env in enumerate(self.test_envs):
            start_time = time.time()
            key_prefix = f"test/{mode}/{seq_idx}/{test_env.name}"
            one_hot_vec = create_one_hot_vec(n_tasks, test_env.task_id)

            self.on_test_start(seq_idx)

            bandit = ExpWeights(arms=list(range(n_arms)), lr=lr, decay=decay, greedy=greedy_bandit, epsilon=epsilon)
            for j in range(num_episodes):

                obs, _ = test_env.reset()
                done, episode_return, episode_len = False, 0, 0
                # Initialize a dictionary to count the number of times each action is selected
                action_counts = {i: 0 for i in range(num_actions)}
                logs_episode_return, logs_episode_num_frames, iter_episode = 0, 0, 0
                while not done:
                    if iter_episode % bandit_step == 0:
                        idx = bandit.sample()
                        one_hot_vec = create_one_hot_vec(n_tasks, idx)
                    arm[test_env.task_id, j, iter_episode] = idx
                    bandit_p[test_env.task_id, :, j, iter_episode] = bandit.p
                    action = self.get_action_test(tf.convert_to_tensor(obs),
                                                  tf.convert_to_tensor(one_hot_vec, dtype=tf.dtypes.float32),
                                                  tf.constant(deterministic))
                    nextobs, reward, done, _, _ = test_env.step(
                        action
                    )
                    episode_return += reward
                    episode_len += 1
                    test_env.render(mode="human")

                    # Increment the count of the selected action
                    action_counts[action] += 1

                    # get feedback for each arm - because we can easily.
                    # We are comparing the main Q val to a fixed Q target which is chosen byt he bandit
                    scores = []

                    # next_actions, _, _ = self.policy_net(torch.Tensor(nextobs).to(self.device).unsqueeze(0), argmax=True)
                    # next_actions = self.get_action_test(tf.convert_to_tensor(nextobs),
                    #                           tf.convert_to_tensor(one_hot_vec, dtype=tf.dtypes.float32),
                    #                           tf.constant(deterministic))
                    q_target = self.critic1(tf.convert_to_tensor([nextobs]),
                                            tf.convert_to_tensor([one_hot_vec], dtype=tf.dtypes.float32))
                    # q_target = next_actions_probs.gather(1, next_actions)
                    q_target = tf.stop_gradient(q_target)
                    value_target = reward + (1.0 - done) * self.gamma * q_target
                    for k in range(n_arms):
                        one_hot_vec = create_one_hot_vec(n_tasks, k)
                        # iterate through the arms/heads to get feedback for the bandit
                        # Don't need to reset the agent with idx as it is not used, until the next round
                        # state_action_values = action_probs.gather(1, torch.Tensor(np.array([action])).long().view(1, -1).to(self.device))
                        state_values = self.critic1(tf.convert_to_tensor([obs]),
                                                    tf.convert_to_tensor([one_hot_vec], dtype=tf.dtypes.float32))

                        # MSE feedback
                        mus_ = state_values.cpu().numpy()
                        # qs[i, y, x] += mus_
                        mse = np.sqrt(np.mean((mus_ - value_target.cpu().numpy())**2))
                        mses[test_env.task_id, k, j, iter_episode] += mse
                        if bandit_loss == 'nll':
                            raise NotImplementedError
                        elif bandit_loss == 'mse':
                            scores.append(min(1 / mse, 50))
                            feedback[test_env.task_id, k, j, iter_episode] = mse
                        else:
                            raise ValueError

                        # x, y = env.agent_pos
                        # freq[y, x] += 1
                # Log the number of times each action was selected
                actions_dict = {f"{key_prefix}/actions/{i}": action_counts[i] for i in range(num_actions)}
                self.logger.store({
                    **actions_dict,
                    key_prefix + "/return": episode_return,
                    key_prefix + "/ep_length": episode_len,
                })
                self.logger.store(test_env.get_statistics(key_prefix))
                total_action_counts = {i: total_action_counts[i] + action_counts[i] for i in range(num_actions)}

            self.on_test_end(seq_idx)
            self.logger.log(f"Finished testing {key_prefix} in {time.time() - start_time:.2f} seconds", color='yellow')

            self.logger.log_tabular(key_prefix + "/return", with_min_and_max=True)
            self.logger.log_tabular(key_prefix + "/ep_length", average_only=True)
            for stat in test_env.get_statistics(key_prefix).keys():
                self.logger.log_tabular(stat, average_only=True)
            for i in range(num_actions):
                self.logger.log_tabular(f"{key_prefix}/actions/{i}", average_only=True)

        # Log the number of times each action was selected across all episodes and test environments
        for i in range(num_actions):
            self.logger.log_tabular(f"test/actions/" + str(i), total_action_counts[i])
