import gym
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.initializers.initializers_v2 import GlorotUniform
from tensorflow.python.keras.layers import Concatenate, Activation, Layer
from typing import Callable, List, Tuple

from cl.sac.models import _choose_head, build_conv_head
from cl.sac.sac import SAC


class VCL_SAC(SAC):
    def __init__(
            self,
            cl_reg_coef: float = 0.0,
            regularize_critic: bool = False,
            first_task_kl: bool = True,
            **vanilla_sac_kwargs
    ) -> None:
        """Variational Continual Learning method. See https://arxiv.org/abs/1710.10628.

        Args:
          cl_reg_coef: Regularization strength for continual learning methods.
            Valid for 'l2', 'ewc', 'mas' continual learning methods.
          regularize_critic: If True, both actor and critic are regularized; if False, only actor
            is regularized.
          first_task_kl: If True, use KL regularization also for the first task in 'vcl'
            continual learning method.
        """
        assert regularize_critic is False, "VCL critic reg not supported"

        super().__init__(**vanilla_sac_kwargs)
        self.cl_reg_coef = cl_reg_coef
        self.regularize_critic = regularize_critic
        self.first_task_kl = first_task_kl
        self.reg_layers = (self.actor.core.layers + self.actor.head_mu.layers)

    def get_auxiliary_loss(self, seq_idx: tf.Tensor) -> tf.Tensor:
        aux_loss = self._regularize(seq_idx, regularize_last_layer=self.first_task_kl)
        aux_loss_coef = tf.cond(
            seq_idx > 0 or self.first_task_kl, lambda: self.cl_reg_coef, lambda: 0.0
        )
        aux_loss *= aux_loss_coef

        return aux_loss

    def on_task_start(self, current_task_idx: int) -> None:
        if current_task_idx > 0:
            self._update_model_prior()

    def _update_model_prior(self) -> None:
        """Update the prior distribution of parameters for the whole network."""

        for layer in self.reg_layers:
            if isinstance(layer, BayesianDense):
                self._update_layer_prior(layer)
            if isinstance(layer, LayerNormalization):
                layer.trainable = False

    @staticmethod
    def _update_layer_prior(layer: Layer) -> None:
        """Update the prior distribution of parameters in the traversed layer."""

        if isinstance(layer, BayesianDense) and layer.num_heads == 1:
            layer.prior_w_mean.assign(layer.posterior_w_mean)
            layer.prior_w_logvar.assign(layer.posterior_w_logvar)

            layer.prior_b_mean.assign(layer.posterior_b_mean)
            layer.prior_b_logvar.assign(layer.posterior_b_logvar)

    def _regularize(self, seq_idx: int, regularize_last_layer: bool) -> tf.Tensor:
        """Calculate the KL loss regularizing the distribution of the parameters for the current
        task to stay close to the distribution of parameters for the previous tasks."""

        kl_loss = 0.0
        for layer in self.reg_layers:
            if isinstance(layer, BayesianDense):
                if layer.num_heads > 1:  # Last layer
                    if not regularize_last_layer:
                        continue
                    input_dim = tf.shape(layer.posterior_w_mean)[0]
                    new_shape = (input_dim, -1, layer.num_heads)
                    kl_loss += kl_divergence(
                        tf.reshape(layer.posterior_w_mean, new_shape)[:, :, seq_idx],
                        tf.reshape(layer.posterior_w_logvar, new_shape)[:, :, seq_idx],
                        tf.reshape(layer.prior_w_mean, new_shape)[:, :, seq_idx],
                        tf.reshape(layer.prior_w_logvar, new_shape)[:, :, seq_idx],
                    )
                    kl_loss += kl_divergence(
                        tf.reshape(layer.posterior_b_mean, (-1, layer.num_heads))[:, seq_idx],
                        tf.reshape(layer.posterior_b_logvar, (-1, layer.num_heads))[:, seq_idx],
                        tf.reshape(layer.prior_b_mean, (-1, layer.num_heads))[:, seq_idx],
                        tf.reshape(layer.prior_b_logvar, (-1, layer.num_heads))[:, seq_idx],
                    )
                else:
                    kl_loss += kl_divergence(
                        layer.posterior_w_mean,
                        layer.posterior_w_logvar,
                        layer.prior_w_mean,
                        layer.prior_w_logvar,
                    )
                    kl_loss += kl_divergence(
                        layer.posterior_b_mean,
                        layer.posterior_b_logvar,
                        layer.prior_b_mean,
                        layer.prior_b_logvar,
                    )

        return kl_loss


class BayesianDense(Layer):
    """Bayesian network implementation of dense (linear) layers. We encode each parameter in the
    layer as a normal distribution."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            activation: Callable = None,
            enable_kl: bool = True,
            num_heads: int = 1,
    ) -> None:
        super().__init__()

        logvar_init = tf.constant_initializer(-6.0)
        w_init = GlorotUniform()
        b_init = tf.zeros_initializer()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.posterior_w_mean = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )
        self.posterior_w_logvar = tf.Variable(
            initial_value=logvar_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )

        self.posterior_b_mean = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )
        self.posterior_b_logvar = tf.Variable(
            initial_value=logvar_init(shape=(output_dim,), dtype="float32"), trainable=True
        )

        self.prior_w_mean = tf.Variable(tf.zeros_like(self.posterior_w_mean), trainable=False)
        self.prior_w_logvar = tf.Variable(tf.zeros_like(self.posterior_w_logvar), trainable=False)

        self.prior_b_mean = tf.Variable(tf.zeros_like(self.posterior_b_mean), trainable=False)
        self.prior_b_logvar = tf.Variable(tf.zeros_like(self.posterior_b_logvar), trainable=False)

        self.activation = activation
        self.enable_kl = enable_kl

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        eps_w = tf.random.normal((self.input_dim, self.output_dim), 0, 1, dtype=tf.float32)
        eps_b = tf.random.normal((1, self.output_dim), 0, 1, dtype=tf.float32)

        weights = eps_w * tf.exp(0.5 * self.posterior_w_logvar) + self.posterior_w_mean
        biases = eps_b * tf.exp(0.5 * self.posterior_b_logvar) + self.posterior_b_mean
        output = tf.matmul(inputs, weights) + biases

        if self.activation is not None:
            output = self.activation(output)

        return output


def variational_mlp(state_shape: Tuple[int], num_tasks: int, hidden_sizes: Tuple[int], activation: Callable,
                    use_layer_norm: bool = False, use_lstm: bool = False) -> Model:
    task_input = Input(shape=num_tasks, name='task_input', dtype=tf.float32)
    conv_in = Input(shape=state_shape, name='conv_head_in')
    conv_head = build_conv_head(conv_in, use_lstm)

    model = Concatenate()([conv_head, task_input])
    model = BayesianDense(model.shape[-1], hidden_sizes[0])(model)
    if use_layer_norm:
        model = LayerNormalization()(model)
        model = Activation(tf.nn.tanh)(model)
    else:
        model = Activation(activation)(model)
    for layer_idx in range(1, len(hidden_sizes)):
        prev_size, next_size = hidden_sizes[layer_idx - 1], hidden_sizes[layer_idx]
        model = BayesianDense(prev_size, next_size, activation=activation)(model)
    model = Model(inputs=[conv_in, task_input], outputs=model)
    return model


class VclMlpActor(Model):
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
        super(VclMlpActor, self).__init__()

        self.num_heads = num_heads
        self.hide_task_id = hide_task_id

        self.core = variational_mlp(state_space.shape, num_tasks, hidden_sizes, activation, use_layer_norm, use_lstm)

        self.head_mu = Sequential(
            [
                InputLayer(input_shape=(hidden_sizes[-1],)),
                BayesianDense(
                    hidden_sizes[-1], action_space.n * num_heads, num_heads=num_heads
                ),
            ]
        )
        self.action_space = action_space

    @property
    def common_variables(self) -> List[tf.Variable]:
        return self.core.trainable_variables + self.head_mu.trainable_variables

    def call(self, obs: tf.Tensor, one_hot_task_id: tf.Tensor, samples_num: int = 1) -> Tuple[tf.Tensor]:
        mus = []
        for sample_idx in range(samples_num):
            x = self.core((obs, one_hot_task_id))
            mu = self.head_mu(x)

            if self.num_heads > 1:
                mu = _choose_head(mu, self.num_heads, one_hot_task_id)

            mus += [mu]

        mu = tf.reduce_mean(tf.stack(mus), 0)

        return mu


def kl_divergence(
        posterior_mean: tf.Tensor,
        posterior_logvar: tf.Tensor,
        prior_mean: tf.Tensor,
        prior_logvar: tf.Tensor,
) -> tf.Tensor:
    numel = tf.cast(tf.size(posterior_mean), tf.float32)
    const_term = -0.5 * numel
    log_std_diff = 0.5 * tf.reduce_sum(prior_logvar - posterior_logvar)

    posterior_var = tf.exp(posterior_logvar)
    prior_var = tf.exp(prior_logvar)

    mu_diff_term = 0.5 * tf.reduce_sum(
        (posterior_var + (posterior_mean - prior_mean)**2) / prior_var
    )
    kl = const_term + log_std_diff + mu_diff_term
    return kl
