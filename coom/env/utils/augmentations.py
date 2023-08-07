import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def random_shift(imgs, pad=4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    batch_size, _, h, w = imgs.shape.as_list()
    imgs = tf.pad(imgs, paddings=[[0, 0], [0, 0], [pad, pad], [pad, pad]], mode='SYMMETRIC')
    imgs = tf.transpose(imgs, perm=[0, 2, 3, 1])
    imgs = tf.image.random_crop(imgs, size=(batch_size, h, w, _))
    imgs = tf.transpose(imgs, perm=[0, 3, 1, 2])
    return imgs


def random_conv(obs: np.ndarray) -> np.ndarray:
    """Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
    weights = tf.random.normal((3, 3, 3, 3))
    obs = tf.pad(obs / 255., paddings=[[1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
    obs = tf.cast(obs, tf.float32)
    obs = tf.expand_dims(obs, axis=0)
    obs = tf.nn.sigmoid(tf.nn.conv2d(obs, weights, strides=[1, 1, 1, 1], padding='VALID'))
    obs = tf.squeeze(obs)
    return obs.numpy() * 255.
