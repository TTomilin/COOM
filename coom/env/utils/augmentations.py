import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def random_shift(obs, pad=16):
    """Vectorized random shift, obs: (H,W,C), pad: #pixels"""
    orig_shape = obs.shape
    obs = tf.pad(obs, paddings=[[pad, pad], [pad, pad], [0, 0]], mode='SYMMETRIC')
    obs = tf.image.random_crop(obs, size=orig_shape).numpy()
    return obs


def random_conv(obs: np.ndarray) -> np.ndarray:
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    weights = tf.random.normal((3, 3, 3, 3))
    obs = tf.pad(obs / 255., paddings=[[1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
    obs = tf.cast(obs, tf.float32)
    obs = tf.expand_dims(obs, axis=0)
    obs = tf.nn.sigmoid(tf.nn.conv2d(obs, weights, strides=[1, 1, 1, 1], padding='VALID'))
    obs = tf.squeeze(obs)
    return obs.numpy() * 255.


def random_noise(obs: np.ndarray) -> np.ndarray:
    """Adds uniform random noise to the image"""
    noise = tf.random.normal(obs.shape, mean=0, stddev=0.1)
    obs = tf.clip_by_value(obs / 255. + noise, 0, 1)
    return obs.numpy() * 255.


def display_aug(obs: np.ndarray, augmented: np.ndarray) -> None:
    """Compares the augmented image with the original one"""
    plt.subplot(1, 2, 1)
    plt.imshow(obs)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(augmented)
    plt.title("Augmented Image")
    plt.show()
