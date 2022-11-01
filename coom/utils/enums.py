from enum import Enum

from coom.env.scenario.arms_dealer import ArmsDealer
from coom.env.scenario.defend_the_center import DefendTheCenter
from coom.env.scenario.parkour import Parkour
from coom.env.scenario.seek_and_slay import SeekAndSlay
from coom.env.scenario.chainsaw import Chainsaw
from coom.env.scenario.dodge_projectiles import DodgeProjectiles
from coom.env.scenario.floor_is_lava import FloorIsLava
from coom.env.scenario.health_gathering import HealthGathering
from coom.env.scenario.hide_and_seek import HideAndSeek
from coom.env.scenario.raise_the_roof import RaiseTheRoof


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"
    PRIORITIZED = "prioritized"


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
    PARKOUR = Parkour
