from typing import Callable, Iterable, List, Tuple

import gym
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, Activation, Concatenate, LSTM, TimeDistributed


def mlp(state_shape: Tuple[int], num_tasks: int, hidden_sizes: Iterable[int], activation: Callable,
        use_layer_norm: bool = False, use_lstm: bool = False, hide_task_id: bool = False) -> Model:
    task_input = Input(shape=num_tasks, name='task_input', dtype=tf.float32)
    conv_in = Input(shape=state_shape, name='conv_head_in')
    conv_head = build_conv_head(conv_in, use_lstm)

    model = conv_head if hide_task_id else Concatenate()([conv_head, task_input])
    model = Dense(hidden_sizes[0])(model)
    if use_layer_norm:
        model = LayerNormalization()(model)
        model = Activation(tf.nn.tanh)(model)
    else:
        model = Activation(activation)(model)
    for size in hidden_sizes[1:]:
        model = Dense(size, activation=activation)(model)
    inputs = conv_in if hide_task_id else [conv_in, task_input]
    model = Model(inputs=inputs, outputs=model)
    return model


def build_conv_head(conv_head, use_lstm):
    for filters, kernel, stride in zip((32, 64, 64), (8, 4, 3), (4, 2, 1)):
        conv_layer = Conv2D(filters, kernel, stride, activation="relu")
        conv_head = TimeDistributed(conv_layer)(conv_head) if use_lstm else conv_layer(conv_head)
    if use_lstm:
        conv_head = TimeDistributed(Flatten())(conv_head)
        conv_head = LSTM(512, activation='tanh')(conv_head)
    else:
        conv_head = Flatten()(conv_head)
    return conv_head


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
        use_lstm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpActor, self).__init__()
        self.num_heads = num_heads
        # if True, one-hot encoding of the task will not be appended to observation.
        self.hide_task_id = hide_task_id

        self.core = mlp(state_space.shape, num_tasks, hidden_sizes, activation, use_layer_norm, use_lstm, hide_task_id)
        self.head_mu = Sequential(
            [
                InputLayer(input_shape=(hidden_sizes[-1],)),
                Dense(action_space.n * num_heads),
            ]
        )
        self.action_space = action_space

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> tf.Tensor:
        logits = self.core(obs) if self.hide_task_id else self.core([obs, one_hot_task_id])
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
        use_lstm: bool = False,
        num_heads: int = 1,
        hide_task_id: bool = False,
    ) -> None:
        super(MlpCritic, self).__init__()
        self.hide_task_id = hide_task_id
        self.num_heads = (
            num_heads  # if True, one-hot encoding of the task will not be appended to observation.
        )

        self.core = mlp(state_space.shape, num_tasks, hidden_sizes, activation, use_layer_norm, use_lstm, hide_task_id)
        self.head = Sequential(
            [InputLayer(input_shape=(hidden_sizes[-1],)), Dense(num_heads * action_space.n)]
        )

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor) -> tf.Tensor:
        logits = self.core(obs) if self.hide_task_id else self.core([obs, one_hot_task_id])
        value = self.head(logits)
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
