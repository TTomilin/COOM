import numpy as np

# Default colors
# Road:         [102, 102, 102]
# Background    [102, 204, 102]
# Grass         [102, 230, 102]
# Car Hull      [0.8, 0.0, 0.0]

domains = {
    'default': {
        'road_color': np.array([102, 102, 102]),
        'bg_color': np.array([102, 204, 102]),
        'grass_color': np.array([102, 230, 102]),
        'car_color': np.array([0.8, 0.0, 0.0]),
    },
    'B1': {
        'road_color': np.array([25, 100, 40]),
        'bg_color': np.array([100, 100, 190]),
        'grass_color': np.array([65, 85, 150]),
        'car_color': np.array([0.5, 0.1, 0.2])
    },
    'B2': {
        'road_color': np.array([162, 102, 102]),
        'bg_color': np.array([200, 200, 20]),
        'grass_color': np.array([220, 220, 20]),
        'car_color': np.array([0.0, 0.0, 0.8])
    },
    'B3': {
        'road_color': np.array([102, 102, 182]),
        'bg_color': np.array([200, 110, 150]),
        'grass_color': np.array([220, 140, 170]),
        'car_color': np.array([0.0, 0.6, 0.3])
    }
}
