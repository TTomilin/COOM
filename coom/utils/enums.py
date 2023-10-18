from enum import Enum

from coom.env.scenario.arms_dealer import ArmsDealer
from coom.env.scenario.chainsaw import Chainsaw
from coom.env.scenario.floor_is_lava import FloorIsLava
from coom.env.scenario.health_gathering import HealthGathering
from coom.env.scenario.hide_and_seek import HideAndSeek
from coom.env.scenario.parkour import Parkour
from coom.env.scenario.pitfall import Pitfall
from coom.env.scenario.raise_the_roof import RaiseTheRoof
from coom.env.scenario.run_and_gun import RunAndGun
from coom.env.utils.augmentations import random_conv, random_shift, random_noise


class DoomScenario(Enum):
    HEALTH_GATHERING = {'class': HealthGathering,
                        'timeout': 2500,
                        'kwargs': ['reward_frame_survived', 'reward_health_hg', 'penalty_health_hg']}
    RUN_AND_GUN = {'class': RunAndGun,
                   'timeout': 1250,
                   'kwargs': ['reward_scaler_traversal', 'reward_kill_rag']}
    CHAINSAW = {'class': Chainsaw,
                'timeout': 2500,
                'kwargs': ['reward_scaler_traversal', 'reward_kill_chain']}
    RAISE_THE_ROOF = {'class': RaiseTheRoof,
                      'timeout': 2500,
                      'kwargs': ['reward_scaler_traversal', 'reward_frame_survived', 'reward_switch_pressed']}
    FLOOR_IS_LAVA = {'class': FloorIsLava,
                     'timeout': 2500,
                     'kwargs': ['reward_scaler_traversal', 'reward_on_platform', 'reward_platform_reached',
                                'reward_frame_survived', 'penalty_lava']}
    HIDE_AND_SEEK = {'class': HideAndSeek,
                     'timeout': 2500,
                     'kwargs': ['reward_scaler_traversal', 'reward_health_has', 'reward_frame_survived',
                                'penalty_health_has']}
    ARMS_DEALER = {'class': ArmsDealer,
                   'timeout': 1000,
                   'kwargs': ['reward_scaler_traversal', 'reward_weapon_ad', 'reward_delivery', 'penalty_passivity']}
    PARKOUR = {'class': Parkour,
               'timeout': 1000,
               'kwargs': ['reward_scaler_traversal']}
    PITFALL = {'class': Pitfall,
               'timeout': 1000,
               'kwargs': ['reward_platform_reached', 'reward_scaler_pitfall', 'penalty_death']}


class Sequence(Enum):
    CD4 = {'scenarios': [DoomScenario.RUN_AND_GUN], 'envs': ['default', 'red', 'blue', 'shadows']}
    CD8 = {'scenarios': [DoomScenario.RUN_AND_GUN],
           'envs': ['obstacles', 'green', 'resized', 'invulnerable', 'default', 'red', 'blue', 'shadows']}
    CO4 = {'scenarios': [DoomScenario.CHAINSAW, DoomScenario.RAISE_THE_ROOF, DoomScenario.RUN_AND_GUN,
                         DoomScenario.HEALTH_GATHERING], 'envs': ['default']}
    CO8 = {'scenarios': [DoomScenario.PITFALL, DoomScenario.ARMS_DEALER, DoomScenario.HIDE_AND_SEEK,
                         DoomScenario.FLOOR_IS_LAVA, DoomScenario.CHAINSAW, DoomScenario.RAISE_THE_ROOF,
                         DoomScenario.RUN_AND_GUN, DoomScenario.HEALTH_GATHERING], 'envs': ['default']}
    COC = {'scenarios': [DoomScenario.PITFALL, DoomScenario.ARMS_DEALER, DoomScenario.HIDE_AND_SEEK,
                         DoomScenario.FLOOR_IS_LAVA, DoomScenario.CHAINSAW, DoomScenario.RAISE_THE_ROOF,
                         DoomScenario.RUN_AND_GUN, DoomScenario.HEALTH_GATHERING], 'envs': ['hard']}
    CD16 = {'scenarios': [DoomScenario.RUN_AND_GUN], 'envs': CD8['envs'] + CD8['envs']}
    CO16 = {'scenarios': CO8['scenarios'] + CO8['scenarios'], 'envs': ['default']}


class Augmentation(Enum):
    CONV = {'method': random_conv}
    SHIFT = {'method': random_shift}
    NOISE = {'method': random_noise}
