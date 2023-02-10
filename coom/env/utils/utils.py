from collections import deque

from scipy import spatial
from vizdoom import ScreenResolution

resolutions = {'800x600': ScreenResolution.RES_800X600,
               '640x480': ScreenResolution.RES_640X480,
               '320x240': ScreenResolution.RES_320X240,
               '160x120': ScreenResolution.RES_160X120}


def default_action_space():
    actions = []
    t_left_right = [[False, False], [False, True], [True, False]]
    m_forward = [[False], [True]]
    execute = [[False], [True]]
    for t in t_left_right:
        for m in m_forward:
            for e in execute:
                actions.append(t + m + e)
    return actions


def get_screen_resolution(resolution: str) -> ScreenResolution:
    if resolution not in resolutions:
        raise ValueError(f'Invalid resolution: {resolution}')
    return resolutions[resolution]


def distance_traversed(game_var_buf: deque, x_index: int, y_index: int) -> float:
    coordinates_curr = [game_var_buf[-1][x_index],
                        game_var_buf[-1][y_index]]
    coordinates_past = [game_var_buf[0][x_index],
                        game_var_buf[0][y_index]]
    return spatial.distance.euclidean(coordinates_curr, coordinates_past)
