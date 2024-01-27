import numpy as np
from scipy.signal import convolve


def random_shift(obs: np.ndarray, pad=4):
    """Vectorized random shift using NumPy, obs: (H,W,C), pad: #pixels"""
    orig_shape = obs.shape
    obs = np.pad(obs, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    crop_x = np.random.randint(0, pad * 2)
    crop_y = np.random.randint(0, pad * 2)
    obs = obs[crop_x:crop_x + orig_shape[0], crop_y:crop_y + orig_shape[1]]
    return obs


def random_conv(obs: np.ndarray, aug_prob=0.5):
    """Applies a random conv2d using NumPy, deviates slightly from https://arxiv.org/abs/1910.05396"""
    if np.random.rand() > aug_prob:
        return obs
    weights = np.random.normal(size=(3, 3, 3, 3))
    obs = np.pad(obs / 255., pad_width=((1, 1), (1, 1), (0, 0)), mode='symmetric')
    obs = np.expand_dims(obs, axis=0)
    convolved = np.zeros_like(obs)
    for i in range(3):  # Assuming obs has 3 channels
        for j in range(3):  # Assuming weights have 3 input channels
            convolved[0, :, :, i] += convolve(obs[0, :, :, j], weights[j, :, :, i], mode='valid')
    obs = convolved
    obs = np.squeeze(obs)
    obs = np.clip(1 / (1 + np.exp(-obs)), 0, 1)  # Sigmoid
    return obs * 255.


def random_noise(obs: np.ndarray):
    """Adds uniform random noise to the image using NumPy"""
    noise = np.random.normal(0, 0.1, obs.shape)
    obs = np.clip(obs / 255. + noise, 0, 1)
    return obs * 255.
