from enum import Enum

from coom.env.base.arms_dealer import ArmsDealer
from coom.env.base.defend_the_center import DefendTheCenter
from coom.env.base.seek_and_slay import SeekAndSlay
from coom.env.base.chainsaw import Chainsaw
from coom.env.base.dodge_projectiles import DodgeProjectiles
from coom.env.base.floor_is_lava import FloorIsLava
from coom.env.base.health_gathering import HealthGathering
from coom.env.base.hide_and_seek import HideAndSeek
from coom.env.base.raise_the_roof import RaiseTheRoof


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"


class DoomScenario(Enum):
    DEFEND_THE_CENTER = DefendTheCenter
    HEALTH_GATHERING = HealthGathering
    SEEK_AND_SLAY = SeekAndSlay
    DODGE_PROJECTILES = DodgeProjectiles
    CHAINSAW = Chainsaw
    RAISE_THE_ROOF = RaiseTheRoof
    FLOOR_IS_LAVA = FloorIsLava
    HIDE_AND_SEEK = HideAndSeek
    ARMS_DEALER = ArmsDealer
