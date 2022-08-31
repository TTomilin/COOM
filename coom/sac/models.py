from typing import Callable, Iterable, List, Tuple

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, LayerNormalization, Concatenate

EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def gaussian_likelihood(x: tf.Tensor, mu: tf.Tensor, log_std: tf.Tensor) -> tf.Tensor:
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(input_tensor=pre_sum, axis=1)


def apply_squashing_func(
    mu: tf.Tensor, pi: tf.Tensor, logp_pi
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    # Adjustment to log prob
    # NOTE: This formula is a little bit magic. To get an understanding of where it
    # comes from, check out the original SAC paper (arXiv 1801.01290) and look in
    # appendix C. This is a more numerically-stable equivalent to Eq 21.
    # Try deriving it yourself as a (very difficult) exercise. :)
    logp_pi -= tf.reduce_sum(input_tensor=2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)

    # Squash those unbounded actions!
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    return mu, pi, logp_pi


def mlp(
        channels: int,
        height: int,
        width: int,
        num_tasks: int,
        hidden_sizes: Tuple[int],
        activation: Callable,
        use_layer_norm: bool = False
) -> Model:
    task_input = Input(shape=num_tasks, name='task_input', dtype=tf.float32)
    conv_in = Input(shape=(height, width, channels), name='conv_head_in')
    conv_head = Conv2D(32, 8, strides=4, activation="relu")(conv_in)
    conv_head = Conv2D(64, 4, strides=2, activation="relu")(conv_head)
    conv_head = Conv2D(64, 3, strides=1, activation="relu")(conv_head)
    conv_head = Flatten()(conv_head)
    model = Concatenate()([conv_head, task_input])
    model = Dense(hidden_sizes[0])(model)
    if use_layer_norm:
        model = LayerNormalization()(model)
        model = Activation(tf.nn.tanh)(model)
    else:
        model = Activation(activation)(model)
    for size in hidden_sizes[1:]:
        model = Dense(size, activation=activation)(model)
    model = Model(inputs=[conv_in, task_input], outputs=model)
    return model


def _choose_head(out: tf.Tensor, num_heads: int, one_hot_task_id: tf.Tensor) -> tf.Tensor:
    """For multi-head output, choose appropriate head.

    We assume that task number is one-hot encoded as a part of observation.

    Args:
      out: multi-head output tensor from the model
      num_heads: number of heads
      one_hot_task_id one-hot encoding of the task

    Returns:
      tf.Tensor: output for the appropriate head
    """
    batch_size = tf.shape(out)[0]
    out = tf.reshape(out, [batch_size, -1, num_heads])
    return tf.squeeze(out @ tf.expand_dims(one_hot_task_id, 2), axis=2)


class MlpActor(Model):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        num_tasks: int,
        hidden_sizes: Tuple[int] = (256, 256),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpActor, self).__init__()
        self.num_heads = num_heads
        # if True, one-hot encoding of the task will not be appended to observation.
        self.hide_task_id = hide_task_id

        self.core = mlp(*state_space.shape, num_tasks, hidden_sizes, activation, use_layer_norm=use_layer_norm)
        self.head_mu = tf.keras.Sequential(
            [
                Input(shape=(hidden_sizes[-1],)),
                Dense(action_space.n * num_heads),
            ]
        )
        self.action_space = action_space

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        obs = tf.transpose(obs, [0, 2, 3, 1])

        logits = self.core([obs, one_hot_task_id])
        mu = self.head_mu(logits)

        if self.num_heads > 1:
            mu = _choose_head(mu, self.num_heads, one_hot_task_id)

        return mu

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Get model parameters which are shared for each task. This excludes head parameters
        in the multi-head setting, as they are separate for each task."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return self.core.trainable_variables + self.head_mu.trainable_variables


class MlpCritic(Model):
    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        num_tasks: int,
        hidden_sizes: Iterable[int] = (256, 256),
        activation: Callable = tf.tanh,
        use_layer_norm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpCritic, self).__init__()
        self.hide_task_id = hide_task_id
        self.num_heads = (
            num_heads  # if True, one-hot encoding of the task will not be appended to observation.
        )

        self.core = mlp(*state_space.shape, num_tasks, hidden_sizes, activation, use_layer_norm=use_layer_norm)
        self.head = tf.keras.Sequential(
            [Input(shape=(hidden_sizes[-1],)), Dense(num_heads * action_space.n)]
        )

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> tf.Tensor:
        obs = tf.transpose(obs, [0, 2, 3, 1])
        value = self.head(self.core([obs, one_hot_task_id]))
        if self.num_heads > 1:
            value = _choose_head(value, self.num_heads, one_hot_task_id)
        return value

    @property
    def common_variables(self) -> List[tf.Variable]:
        """Get model parameters which are shared for each task. This excludes head parameters
        in the multi-head setting, as they are separate for each task."""
        if self.num_heads > 1:
            return self.core.trainable_variables
        elif self.num_heads == 1:
            return self.core.trainable_variables + self.head.trainable_variables


class PopArtMlpCritic(MlpCritic):
    """PopArt implementation.

    PopArt is a method for normalizing returns, especially useful in multi-task learning.
    See https://arxiv.org/abs/1602.07714 and https://arxiv.org/abs/1809.04474v1.
    """

    def __init__(self, beta=3e-4, **kwargs) -> None:
        super(PopArtMlpCritic, self).__init__(**kwargs)

        self.moment1 = tf.Variable(tf.zeros((self.num_heads, 1)), trainable=False)
        self.moment2 = tf.Variable(tf.ones((self.num_heads, 1)), trainable=False)
        self.sigma = tf.Variable(tf.ones((self.num_heads, 1)), trainable=False)

        self.beta = beta

    @tf.function
    def unnormalize(self, x: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
        # TODO Rewrite
        moment1 = tf.squeeze(obs[:, -self.num_heads :] @ self.moment1, axis=1)
        sigma = tf.squeeze(obs[:, -self.num_heads :] @ self.sigma, axis=1)
        return x * sigma + moment1

    @tf.function
    def normalize(self, x: tf.Tensor, obs: tf.Tensor) -> tf.Tensor:
        # TODO Rewrite
        moment1 = tf.squeeze(obs[:, -self.num_heads :] @ self.moment1, axis=1)
        sigma = tf.squeeze(obs[:, -self.num_heads :] @ self.sigma, axis=1)
        return (x - moment1) / sigma

    @tf.function
    def update_stats(self, returns: tf.Tensor, obs: tf.Tensor) -> None:
        # TODO Rewrite
        task_counts = tf.reduce_sum(obs[:, -self.num_heads :], axis=0)
        batch_moment1 = tf.reduce_sum(
            tf.expand_dims(returns, 1) * obs[:, -self.num_heads :], axis=0
        ) / tf.math.maximum(task_counts, 1.0)
        batch_moment2 = tf.reduce_sum(
            tf.expand_dims(returns * returns, 1) * obs[:, -self.num_heads :], axis=0
        ) / tf.math.maximum(task_counts, 1.0)

        update_pos = tf.expand_dims(tf.cast(task_counts > 0, tf.float32), 1)
        new_moment1 = self.moment1 + update_pos * (
            self.beta * (tf.expand_dims(batch_moment1, 1) - self.moment1)
        )
        new_moment2 = self.moment2 + update_pos * (
            self.beta * (tf.expand_dims(batch_moment2, 1) - self.moment2)
        )
        new_sigma = tf.math.sqrt(new_moment2 - new_moment1 * new_moment1)
        new_sigma = tf.clip_by_value(new_sigma, 1e-4, 1e6)

        # Update weights of the last layer.
        last_layer = self.head.layers[-1]
        last_layer.kernel.assign(
            last_layer.kernel * tf.transpose(self.sigma) / tf.transpose(new_sigma)
        )
        last_layer.bias.assign(
            (last_layer.bias * tf.squeeze(self.sigma) + tf.squeeze(self.moment1 - new_moment1))
            / tf.squeeze(new_sigma)
        )

        self.moment1.assign(new_moment1)
        self.moment2.assign(new_moment2)
        self.sigma.assign(new_sigma)
