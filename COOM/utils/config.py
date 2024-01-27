from enum import Enum

from COOM.env.scenarios.arms_dealer.arms_dealer import ArmsDealer
from COOM.env.scenarios.chainsaw.chainsaw import Chainsaw
from COOM.env.scenarios.floor_is_lava.floor_is_lava import FloorIsLava
from COOM.env.scenarios.health_gathering.health_gathering import HealthGathering
from COOM.env.scenarios.hide_and_seek.hide_and_seek import HideAndSeek
from COOM.env.scenarios.pitfall.pitfall import Pitfall
from COOM.env.scenarios.raise_the_roof.raise_the_roof import RaiseTheRoof
from COOM.env.scenarios.run_and_gun.run_and_gun import RunAndGun
from COOM.utils.augmentations import random_conv, random_shift, random_noise


class Augmentation(Enum):
    CONV = random_conv
    SHIFT = random_shift
    NOISE = random_noise


class Sequence(Enum):
    CD4 = 1
    CD8 = 2
    CO4 = 3
    CO8 = 4
    CD16 = 5
    CO16 = 6
    COC = 7
    MIXED = 8


class Scenario(Enum):
    PITFALL = 1
    ARMS_DEALER = 2
    FLOOR_IS_LAVA = 3
    HIDE_AND_SEEK = 4
    CHAINSAW = 5
    RAISE_THE_ROOF = 6
    RUN_AND_GUN = 7
    HEALTH_GATHERING = 8


scenario_config = {
    Scenario.PITFALL: {
        'class': Pitfall,
        'args': ['reward_platform_reached', 'reward_scaler_pitfall', 'penalty_death']
    },
    Scenario.ARMS_DEALER: {
        'class': ArmsDealer,
        'args': ['reward_scaler_traversal', 'reward_weapon_ad', 'reward_delivery', 'penalty_passivity']
    },
    Scenario.FLOOR_IS_LAVA: {
        'class': FloorIsLava,
        'args': ['reward_scaler_traversal', 'reward_on_platform', 'reward_platform_reached',
                 'reward_frame_survived', 'penalty_lava']
    },
    Scenario.HIDE_AND_SEEK: {
        'class': HideAndSeek,
        'args': ['reward_scaler_traversal', 'reward_health_has', 'reward_frame_survived',
                 'penalty_health_has']
    },
    Scenario.CHAINSAW: {
        'class': Chainsaw,
        'args': ['reward_scaler_traversal', 'reward_kill_chain']
    },
    Scenario.RAISE_THE_ROOF: {
        'class': RaiseTheRoof,
        'args': ['reward_scaler_traversal', 'reward_frame_survived', 'reward_switch_pressed']
    },
    Scenario.RUN_AND_GUN: {
        'class': RunAndGun,
        'args': ['reward_scaler_traversal', 'reward_kill_rag']
    },
    Scenario.HEALTH_GATHERING: {
        'class': HealthGathering,
        'args': ['reward_frame_survived', 'reward_health_hg', 'penalty_health_hg']
    },
}

CD_scenarios = [Scenario.RUN_AND_GUN]
CO_scenarios = [Scenario.PITFALL, Scenario.ARMS_DEALER, Scenario.HIDE_AND_SEEK, Scenario.FLOOR_IS_LAVA,
                Scenario.CHAINSAW, Scenario.RAISE_THE_ROOF, Scenario.RUN_AND_GUN,
                Scenario.HEALTH_GATHERING]
CD_tasks = ['obstacles', 'green', 'resized', 'monsters', 'default', 'red', 'blue', 'shadows']
CO_tasks = ['default']
COC_tasks = ['hard']

sequence_scenarios = {
    Sequence.CD4: CD_scenarios,
    Sequence.CO4: CO_scenarios[4:],
    Sequence.CD8: CD_scenarios,
    Sequence.CO8: CO_scenarios,
    Sequence.CD16: CD_scenarios,
    Sequence.CO16: CO_scenarios + CO_scenarios,
    Sequence.COC: CO_scenarios,
    Sequence.MIXED: [item for pair in zip(CD_scenarios * len(CO_scenarios), CO_scenarios) for item in pair]
}

sequence_tasks = {
    Sequence.CD4: CD_tasks[4:],
    Sequence.CO4: CO_tasks,
    Sequence.CD8: CD_tasks,
    Sequence.CO8: CO_tasks,
    Sequence.CD16: CD_tasks + CD_tasks,
    Sequence.CO16: CO_tasks,
    Sequence.COC: COC_tasks,
    Sequence.MIXED: [item for pair in zip(CD_tasks, CO_tasks * len(CD_tasks)) for item in pair]
}


default_wrapper_config = {
    'augment': False,
    'augmentation': 'conv',
    'resize': True,
    'frame_height': 84,
    'frame_width': 84,
    'rescale': True,
    'normalize_observation': True,
    'frame_stack': 4,
    'lstm': False,
    'record': False,
    'record_dir': 'videos',
    'sparse_rewards': False,
}
