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


class BufferType(Enum):
    FIFO = "fifo"
    RESERVOIR = "reservoir"
    PRIORITIZED = "prioritized"


class DoomScenario(Enum):
    HEALTH_GATHERING = HealthGathering
    RUN_AND_GUN = RunAndGun
    CHAINSAW = Chainsaw
    RAISE_THE_ROOF = RaiseTheRoof
    FLOOR_IS_LAVA = FloorIsLava
    HIDE_AND_SEEK = HideAndSeek
    ARMS_DEALER = ArmsDealer
    PARKOUR = Parkour
    PITFALL = Pitfall
