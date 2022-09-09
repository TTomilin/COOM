from enum import Enum

from coom.doom.env.extended.chainsaw_impl import ChainsawImpl
from coom.doom.env.extended.defend_the_center_impl import DefendTheCenterImpl
from coom.doom.env.extended.dodge_projectiles_impl import DodgeProjectilesImpl
from coom.doom.env.extended.floor_is_lava_impl import FloorIsLavaImpl
from coom.doom.env.extended.health_gathering_impl import HealthGatheringImpl
from coom.doom.env.extended.hide_and_seek_impl import HideAndSeekImpl
from coom.doom.env.extended.raise_the_roof_impl import RaiseTheRoofImpl
from coom.doom.env.extended.seek_and_slay_impl import SeekAndSlayImpl


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"


class DoomScenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenterImpl
    HEALTH_GATHERING = HealthGatheringImpl
    SEEK_AND_SLAY = SeekAndSlayImpl
    DODGE_PROJECTILES = DodgeProjectilesImpl
    CHAINSAW = ChainsawImpl
    RAISE_THE_ROOF = RaiseTheRoofImpl
    FLOOR_IS_LAVA = FloorIsLavaImpl
    HIDE_AND_SEEK = HideAndSeekImpl
