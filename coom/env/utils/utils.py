from collections import deque

from scipy import spatial
from vizdoom import ScreenResolution

resolutions = {800: ScreenResolution.RES_800X600,
               640: ScreenResolution.RES_640X480,
               320: ScreenResolution.RES_320X240,
               160: ScreenResolution.RES_160X120}


def get_screen_resolution(resolution: int) -> ScreenResolution:
    if resolution not in resolutions:
        raise ValueError(f'Invalid resolution: {resolution}')
    return resolutions[resolution]


def distance_traversed(game_var_buf: deque, x_index: int, y_index: int) -> float:
    coordinates_curr = [game_var_buf[-1][x_index],
                        game_var_buf[-1][y_index]]
    coordinates_past = [game_var_buf[0][x_index],
                        game_var_buf[0][y_index]]
    return spatial.distance.euclidean(coordinates_curr, coordinates_past)