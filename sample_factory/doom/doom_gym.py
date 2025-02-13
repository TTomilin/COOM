import copy
import os
import random
import re
import time
from os.path import join
from threading import Thread
from typing import Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from filelock import FileLock, Timeout
from gymnasium.utils import seeding
from vizdoom import AutomapMode, DoomGame, Mode, ScreenResolution

from sample_factory.algo.utils.spaces.discretized import Discretized
from sample_factory.utils.utils import log, project_tmp_dir

resolutions = {'1920x1080': ScreenResolution.RES_1920X1080,
               '1600x1200': ScreenResolution.RES_1600X1200,
               '1280x720': ScreenResolution.RES_1280X720,
               '800x600': ScreenResolution.RES_800X600,
               '640x480': ScreenResolution.RES_640X480,
               '320x240': ScreenResolution.RES_320X240,
               '160x120': ScreenResolution.RES_160X120}


def doom_lock_file(max_parallel):
    """
    Doom instances tend to have problems starting when a lot of them are initialized in parallel.
    This is not a problem during normal execution once the envs are initialized.

    The "sweet spot" for the number of envs that can be initialized in parallel is about 5-10.
    Here we use file locking mechanism to ensure that only a limited amount of envs are being initialized at the same
    time.
    This tends to be more of a problem for multiplayer envs.

    This also has an advantage of working across completely independent process groups, e.g. different experiments.
    """
    lock_filename = f"doom_{random.randrange(0, max_parallel):03d}.lockfile"

    tmp_dir = project_tmp_dir()
    lock_path = join(tmp_dir, lock_filename)
    return lock_path


def key_to_action_default(key):
    """
    MOVE_FORWARD
    MOVE_BACKWARD
    MOVE_RIGHT
    MOVE_LEFT
    SELECT_WEAPON1
    SELECT_WEAPON2
    SELECT_WEAPON3
    SELECT_WEAPON4
    SELECT_WEAPON5
    SELECT_WEAPON6
    SELECT_WEAPON7
    ATTACK
    SPEED
    TURN_LEFT_RIGHT_DELTA
    """
    from pynput.keyboard import Key

    # health gathering
    action_table = {
        Key.left: 0,
        Key.right: 1,
        Key.up: 2,
        Key.down: 3,
    }

    # action_table = {
    #     Key.up: 0,
    #     Key.down: 1,
    #     Key.alt: 6,
    #     Key.ctrl: 11,
    #     Key.shift: 12,
    #     Key.space: 13,
    #     Key.right: 'turn_right',
    #     Key.left: 'turn_left',
    # }

    return action_table.get(key, None)


def get_screen_resolution(resolution: str) -> ScreenResolution:
    if resolution not in resolutions:
        raise ValueError(f'Invalid resolution: {resolution}')
    return resolutions[resolution]


class VizdoomEnv(gym.Env):
    def __init__(
            self,
            config_file: str,
            action_space: gym.Space,
            safety_bound: float,
            unsafe_reward: float,
            timeout: int,
            level=1,
            constraint='soft',
            coord_limits=None,
            max_histogram_length=None,
            show_automap=False,
            use_depth_buffer=False,
            render_depth_buffer=False,
            render_with_bounding_boxes=False,
            segment_objects=False,
            skip_frames=1,
            async_mode=False,
            record_to=None,
            env_modification: str = None,
            resolution: str = None,
            seed: Optional[int] = None,
            render_mode: Optional[str] = None,
    ):
        self.initialized = False

        # essential game data
        self.game = None
        self.curr_seed = 0
        self.rng = None
        self.skip_frames = skip_frames
        self.async_mode = async_mode
        self.timeout = timeout

        # optional - for topdown view rendering and visitation heatmaps
        self.show_automap = show_automap
        self.use_depth_buffer = use_depth_buffer
        self.render_depth_buffer = render_depth_buffer
        self.render_with_bounding_boxes = render_with_bounding_boxes
        self.segment_objects = segment_objects
        self.coord_limits = coord_limits

        self.unique_label_names = set()
        self.object_name_to_color = {
            "DoomPlayer": (255, 255, 255),     # White

            # Weapons
            "Pistol": (64, 64, 64),            # Dark Gray
            "Shotgun": (139, 69, 19),          # Brown
            "SuperShotgun": (160, 82, 45),     # Darker Brown
            "Chaingun": (0, 0, 128),           # Navy Blue
            "RocketLauncher": (128, 0, 0),     # Maroon
            "PlasmaRifle": (0, 128, 128),      # Teal
            "BFG9000": (0, 255, 255),          # Cyan

            # Bonuses and Items
            "ArmorBonus": (0, 255, 0),         # Bright Green
            "BlurSphere": (255, 0, 255),       # Magenta
            "Allmap": (255, 255, 0),           # Yellow
            "Backpack": (153, 102, 51),        # Dark Beige
            "RadSuit": (0, 102, 102),          # Dark Teal
            "Infrared": (255, 165, 0),         # Orange

            # Enemies
            "CacoDemon": (255, 0, 0),          # Bright Red
            "LostSoul": (255, 255, 255),       # White
            "ZombieMan": (128, 128, 128),      # Gray
            "ShotgunGuy": (64, 0, 64),         # Purple
            "ChaingunGuy": (0, 64, 128),       # Dark Cyan
            "DoomImp": (139, 69, 19),          # Saddle Brown
            "Demon": (255, 105, 180),          # Pink
            "Revenant": (210, 180, 140),       # Tan

            # Environmental Hazards
            "ExplosiveBarrel": (255, 0, 0),    # Red

            # Health Items
            "HealthBonus": (0, 255, 0),        # Green
            "Stimpack": (192, 0, 0),           # Crimson
            "Medikit": (255, 0, 0),            # Red

            # Ammo
            "Shell": (255, 255, 0),            # Yellow
            "Cell": (0, 0, 255),               # Blue
            "RocketAmmo": (128, 0, 0),         # Maroon

            # Decorations and Environmental Objects
            "Stalagtite": (128, 128, 128),     # Light Gray
            "TorchTree": (0, 255, 150),        # Light Green
            "BigTree": (0, 200, 100),          # Dark Green
            "Gibs": (100, 200, 150),           # Light Brownish Green
            "BrainStem": (100, 0, 200),        # Purple
            "HeartColumn": (200, 0, 50),       # Deep Red
            "TechPillar": (150, 150, 255),     # Light Blue
            "ShortRedColumn": (255, 50, 50),   # Bright Red
            "ShortGreenColumn": (50, 255, 50), # Bright Green
            "RocketSmokeTrail": (128, 128, 128),# Smoke Gray
            "Rocket": (200, 0, 0),             # Bright Red
            "BulletPuff": (220, 220, 220),     # Light Gray
            "Blood": (150, 0, 0),              # Blood Red
            "GibbedMarine": (180, 50, 50),     # Dark Red
            "TechLamp": (100, 100, 255),       # Light Blue
            "SmallBloodPool": (80, 0, 0),      # Dark Blood Red
        }

        self.object_id_to_color = {
            0: (0, 0, 0),        # Ground & Walls - Black
            1: (128, 128, 128),  # Ceiling - Gray
            255: (255, 255, 255) # Agent - White
        }

        # Assign the same color for everything else
        self.default_color = (200, 200, 200)  # All unknown objects

        # can be adjusted after the environment is created (but before any reset() call) via observation space wrapper
        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.resolution = resolution
        self.calc_observation_space()

        self.black_screen = None

        # provided as a part of environment definition, since these depend on the scenario and
        # can be quite complex multi-discrete spaces
        self.action_space = action_space
        self.composite_action_space = hasattr(self.action_space, "spaces")

        self.delta_actions_scaling_factor = 7.5

        if os.path.isabs(config_file):
            self.config_path = config_file
        else:
            scenarios_dir = join(os.path.dirname(__file__), "scenarios")
            self.config_path = join(scenarios_dir, config_file)
            if not os.path.isfile(self.config_path):
                log.warning(
                    "File %s not found in scenarios dir %s. Consider providing absolute path?",
                    config_file,
                    scenarios_dir,
                )

        base_path = self.config_path.replace('_all', '').replace('.cfg', f'_{level}')
        self.hard_constraint = constraint == 'hard'
        if self.hard_constraint:
            base_path += f'_{constraint}'
        if env_modification:
            base_path += f'_{env_modification}'
        self.scenario_path = f"{base_path}.wad"

        self.variable_indices = self._parse_variable_indices(self.config_path)

        # record full episodes using VizDoom recording functionality
        self.record_to = record_to
        self.curr_demo_dir = None

        self.is_multiplayer = False  # overridden in derived classes

        # (optional) histogram to track positional coverage
        # do not pass coord_limits if you don't need this, to avoid extra calculation
        self.max_histogram_length = max_histogram_length
        self.current_histogram, self.previous_histogram = None, None
        if self.coord_limits:
            x = self.coord_limits[2] - self.coord_limits[0]
            y = self.coord_limits[3] - self.coord_limits[1]
            if x > y:
                len_x = self.max_histogram_length
                len_y = int((y / x) * self.max_histogram_length)
            else:
                len_x = int((x / y) * self.max_histogram_length)
                len_y = self.max_histogram_length
            self.current_histogram = np.zeros((len_x, len_y), dtype=np.int32)
            self.previous_histogram = np.zeros_like(self.current_histogram)

        # helpers for human play with pynput keyboard input
        self._terminate = False
        self._current_actions = []
        self._actions_flattened = None

        self._prev_info = None
        self._last_episode_info = None

        self._num_episodes = 0

        self.safety_bound = 0.0 if self.hard_constraint else safety_bound
        self.unsafe_reward = unsafe_reward

        self.mode = "algo"

        self.render_mode = render_mode
        self.metadata['render_fps'] = 10

        self.seed(seed)

    def seed(self, seed: Optional[int] = None):
        """
        Used to seed the actual Doom env.
        If None is passed, the seed is generated randomly.
        """
        self.rng, self.curr_seed = seeding.np_random(seed=seed)
        self.curr_seed = self.curr_seed % (2 ** 32)  # Doom only supports 32-bit seeds
        return [self.curr_seed, self.rng]

    def calc_observation_space(self):
        self.observation_space = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, self.channels), dtype=np.uint8)

    def _set_game_mode(self, mode):
        if mode == "replay":
            self.game.set_mode(Mode.PLAYER)
        else:
            if self.async_mode:
                log.info("Starting in async mode! Use this only for testing, otherwise PLAYER mode is much faster")
                self.game.set_mode(Mode.ASYNC_PLAYER)
            else:
                self.game.set_mode(Mode.PLAYER)

    def _create_doom_game(self, mode):
        self.game = DoomGame()
        self.game.set_screen_resolution(get_screen_resolution(self.resolution))
        self.game.load_config(self.config_path)
        self.game.set_doom_scenario_path(self.scenario_path)
        self.game.set_seed(self.curr_seed)
        self.game.set_depth_buffer_enabled(self.use_depth_buffer or self.render_depth_buffer)
        self.game.set_labels_buffer_enabled(self.render_with_bounding_boxes or self.segment_objects)

        if mode == "human" or mode == "replay" or self.render_mode == 'human':
            self.game.add_game_args("+freelook 1")
            self.game.set_window_visible(True)
            self.frame_skip = 1
        elif mode == "algo":
            self.game.set_window_visible(False)
        else:
            raise Exception("Unsupported mode")

        self._set_game_mode(mode)

    def _game_init(self, with_locking=True, max_parallel=10):
        lock_file = lock = None
        if with_locking:
            lock_file = doom_lock_file(max_parallel)
            lock = FileLock(lock_file)

        init_attempt = 0
        while True:
            init_attempt += 1
            try:
                if with_locking:
                    with lock.acquire(timeout=20):
                        self.game.init()
                else:
                    self.game.init()

                break
            except Timeout:
                if with_locking:
                    log.debug(
                        "Another process currently holds the lock %s, attempt: %d",
                        lock_file,
                        init_attempt,
                    )
            except Exception as exc:
                log.warning("VizDoom game.init() threw an exception %r. Terminate process...", exc)
                from sample_factory.envs.env_utils import EnvCriticalError

                raise EnvCriticalError()

    def initialize(self):
        self._create_doom_game(self.mode)

        # (optional) top-down view provided by the game engine
        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            # self.game.add_game_args("+am_restorecolors")
            # self.game.add_game_args("+am_followplayer 1")
            background_color = "ffffff"
            self.game.add_game_args("+viz_am_center 1")
            self.game.add_game_args("+am_backcolor " + background_color)
            self.game.add_game_args("+am_tswallcolor dddddd")
            # self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args("+am_yourcolor " + background_color)
            self.game.add_game_args("+am_cheat 0")
            self.game.add_game_args("+am_thingcolor 0000ff")  # player color
            self.game.add_game_args("+am_thingcolor_item 00ff00")
            # self.game.add_game_args("+am_thingcolor_citem 00ff00")

        self._game_init()
        self.initialized = True

    def _ensure_initialized(self):
        if not self.initialized:
            self.initialize()

    @staticmethod
    def _parse_variable_indices(config):
        with open(config, "r") as config_file:
            lines = config_file.readlines()
        lines = [ln.strip() for ln in lines]

        variable_indices = {}

        for line in lines:
            if line.startswith("#"):
                continue  # comment

            variables_syntax = r"available_game_variables[\s]*=[\s]*\{(.*)\}"
            match = re.match(variables_syntax, line)
            if match is not None:
                variables_str = match.groups()[0]
                variables_str = variables_str.strip()
                variables = variables_str.split(" ")
                for i, variable in enumerate(variables):
                    variable_indices[variable] = i
                break

        return variable_indices

    def _black_screen(self):
        if self.black_screen is None:
            self.black_screen = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return self.black_screen

    def _game_variables_dict(self, state):
        game_variables = state.game_variables
        variables = {}
        for variable, idx in self.variable_indices.items():
            variables[variable] = game_variables[idx]
        return variables

    @staticmethod
    def demo_path(episode_idx, record_to):
        demo_name = f"e{episode_idx:03d}.lmp"
        demo_path_ = join(record_to, demo_name)
        demo_path_ = os.path.normpath(demo_path_)
        return demo_path_

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs and kwargs["seed"]:
            self.seed(kwargs["seed"])

        self._ensure_initialized()

        episode_started = False
        if self.record_to and not self.is_multiplayer:
            # does not work in multiplayer (uses different mechanism)
            if not os.path.exists(self.record_to):
                os.makedirs(self.record_to)

            demo_path = self.demo_path(self._num_episodes, self.record_to)
            self.curr_demo_dir = os.path.dirname(demo_path)
            log.warning(f"Recording episode demo to {demo_path=}")

            if len(demo_path) > 101:
                log.error(f"Demo path {len(demo_path)=}>101, will not record demo")
                log.error(
                    "This seems to be a bug in VizDoom, please just use a shorter demo path, i.e. set --record_to to /tmp/doom_recs"
                )
            else:
                self.game.new_episode(demo_path)
                episode_started = True

        if self._num_episodes > 0 and not episode_started:
            # no demo recording (default)
            self.game.new_episode()

        state = self.game.get_state()
        img = self._get_obs_from_state(state)

        info = {}

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        self._actions_flattened = None
        self._last_episode_info = copy.deepcopy(self._prev_info)
        self._prev_info = None

        self._num_episodes += 1

        return img, info  # since Gym 0.26.0, we return the info dict as second return value

    def _convert_actions(self, actions):
        """Convert actions from gym action space to the action space expected by Doom game."""

        if self.composite_action_space:
            # composite action space with multiple subspaces
            spaces = self.action_space.spaces
        else:
            # simple action space, e.g. Discrete. We still treat it like composite of length 1
            spaces = (self.action_space,)
            actions = (actions,)

        actions_flattened = []
        for i, action in enumerate(actions):
            if isinstance(spaces[i], Discretized):
                # discretized continuous action
                # check discretized first because it's a subclass of gym.spaces.Discrete
                # the order of if clauses here matters! DON'T CHANGE THE ORDER OF IFS!

                continuous_action = spaces[i].to_continuous(action)
                actions_flattened.append(continuous_action)
            elif isinstance(spaces[i], gym.spaces.Discrete):
                # standard discrete action
                num_non_idle_actions = spaces[i].n - 1
                action_one_hot = np.zeros(num_non_idle_actions, dtype=np.uint8)
                if action > 0:
                    action_one_hot[action - 1] = 1  # 0th action in each subspace is a no-op

                actions_flattened.extend(action_one_hot)
            elif isinstance(spaces[i], gym.spaces.Box):
                # continuous action
                actions_flattened.extend(list(action * self.delta_actions_scaling_factor))
            else:
                raise NotImplementedError(f"Action subspace type {type(spaces[i])} is not supported!")

        return actions_flattened

    def _vizdoom_variables_bug_workaround(self, info, done):
        """Some variables don't get reset to zero on game.new_episode(). This fixes it (also check overflow?)."""
        if done and "DAMAGECOUNT" in info:
            log.info("DAMAGECOUNT value on done: %r", info.get("DAMAGECOUNT"))

        if self._last_episode_info is not None:
            bugged_vars = ["DEATHCOUNT", "HITCOUNT", "DAMAGECOUNT"]
            for v in bugged_vars:
                if v in info:
                    info[v] -= self._last_episode_info.get(v, 0)

    def _generate_unique_color(self):
        # Generate a unique color not already in use
        while True:
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))
            if color not in self.object_name_to_color.values():
                return color

    def _get_obs_from_state(self, state):
        # Default observation: screen buffer
        obs = state.automap_buffer if self.show_automap else state.screen_buffer
        obs = np.transpose(obs, (1, 2, 0))  # Transpose to HWC format

        # Apply segmentation if enabled
        if self.segment_objects:
            obs = self.segment_obs(obs, state)
        return obs

    def segment_obs(self, obs, state):
        # Convert obs to BGR for OpenCV processing
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        # Get label buffer and labels information
        label_buffer = state.labels_buffer.astype(np.uint8)
        labels = state.labels  # List of Label objects
        # Build mapping from label value to object name or ID
        value_to_object = {}
        for label in labels:
            self.unique_label_names.add(label.object_name)
            value_to_object[label.value] = label.object_name  # or label.object_id
        # Assign consistent colors to each object
        for value, object_name in value_to_object.items():
            if object_name not in self.object_name_to_color:
                # Assign a unique color to each object_name
                self.object_name_to_color[object_name] = self._generate_unique_color()
        # Create segmented observation
        segmented_obs = np.zeros_like(obs)
        unique_values = np.unique(label_buffer)
        for value in unique_values:
            object_name = value_to_object.get(value, None)
            if object_name is not None:
                color = self.object_name_to_color[object_name]
            else:
                color = self.object_id_to_color.get(value, None)
            if color is None:
                color = self.default_color  # For undefined values

            # Create a mask for current value
            mask = (label_buffer == value).astype(np.uint8)

            # Apply color to the segmented observation
            for i in range(3):  # Apply color per channel
                segmented_obs[:, :, i] += (mask * color[i])
        # Update observation with segmented version
        obs = segmented_obs
        return obs

    def _process_game_step(self, state, done, info):
        if not done:
            observation = self._get_obs_from_state(state)
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info_all(game_variables))
            self._update_histogram(info)
            self._prev_info = copy.copy(info)
        else:
            observation = self._black_screen()

            # when done=True Doom does not allow us to call get_info, so we provide info from the last frame
            info.update(self._prev_info)

        self._vizdoom_variables_bug_workaround(info, done)

        return observation, done, info

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Action is either a single value (discrete, one-hot), or a tuple with an action for each of the
        discrete action subspaces.
        """
        if self._actions_flattened is not None:
            # provided externally, e.g. via human play
            actions_flattened = self._actions_flattened
            self._actions_flattened = None
        else:
            actions_flattened = self._convert_actions(actions)

        default_info = {"num_frames": self.skip_frames}
        reward = self.game.make_action(actions_flattened, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        observation, done, info = self._process_game_step(state, done, default_info)

        # Gym 0.26.0 changes
        truncated = self.game.get_episode_time() > self.timeout
        terminated = done
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        mode = self.render_mode
        if mode is None:
            return

        state = self.game.get_state()
        screen = self._get_obs_from_state(state) if state else self._black_screen()

        if self.render_with_bounding_boxes and state:
            self.add_bounding_boxes(screen, state)

        ret = state.depth_buffer if self.render_depth_buffer and state else screen

        if mode == "rgb_array":
            return ret
        elif mode == "human":
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            cv2.imshow("HASARD", screen)

            # Render a separate screen for the depth buffer
            if self.use_depth_buffer and state:
                depth_buffer = state.depth_buffer
                # Normalize depth buffer for better visualization
                normalized_depth = (255 * (depth_buffer / np.max(depth_buffer))).astype(np.uint8)
                cv2.imshow("Depth Buffer", depth_buffer)
                cv2.imshow("Normalized Depth Buffer", normalized_depth)

            cv2.waitKey(1)

        return ret

    def add_bounding_boxes(self, screen, state):
        label_buffer = state.labels_buffer  # Per-pixel object ID
        # Normalize label buffer for processing
        label_buffer = label_buffer.astype(np.uint8)
        # Find unique object IDs in the label buffer
        unique_ids = np.unique(label_buffer)
        # Create a mask for each object and find contours
        for obj_id in unique_ids:
            if obj_id in [0, 1, 255]:  # Skip background, agent
                continue

            # Create a binary mask for the current object
            mask = (label_buffer == obj_id).astype(np.uint8)

            # Find contours of the object
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around the object
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(screen, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

                # Optional: Add object ID as text
                cv2.putText(screen, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)

    def close(self):
        try:
            if self.game is not None:
                self.game.close()
        except RuntimeError as exc:
            log.warning("Runtime error in VizDoom game close(): %r", exc)

    def get_info(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())

        info_dict = {"pos": self.get_positions(variables)}
        info_dict.update(variables)
        return info_dict

    def get_info_all(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())
        info = self.get_info(variables)
        if self.previous_histogram is not None:
            info["previous_histogram"] = copy.deepcopy(self.previous_histogram)
        return info

    def get_positions(self, variables):
        return self._get_positions(variables)

    @staticmethod
    def _get_positions(variables):
        have_coord_data = True
        required_vars = ["POSITION_X", "POSITION_Y", "ANGLE"]
        for required_var in required_vars:
            if required_var not in variables:
                have_coord_data = False
                break

        x = y = a = np.nan
        if have_coord_data:
            x = variables["POSITION_X"]
            y = variables["POSITION_Y"]
            a = variables["ANGLE"]

        return {"agent_x": x, "agent_y": y, "agent_a": a}

    def get_automap_buffer(self):
        if self.game.is_episode_finished():
            return None
        state = self.game.get_state()
        map_ = state.automap_buffer
        map_ = np.swapaxes(map_, 0, 2)
        map_ = np.swapaxes(map_, 0, 1)
        return map_

    def _update_histogram(self, info, eps=1e-8):
        if self.current_histogram is None:
            return
        agent_x, agent_y = info["pos"]["agent_x"], info["pos"]["agent_y"]

        # Get agent coordinates normalized to [0, 1]
        dx = (agent_x - self.coord_limits[0]) / (self.coord_limits[2] - self.coord_limits[0])
        dy = (agent_y - self.coord_limits[1]) / (self.coord_limits[3] - self.coord_limits[1])

        # Rescale coordinates to histogram dimensions
        # Subtract eps to exclude upper bound of dx, dy
        dx = int((dx - eps) * self.current_histogram.shape[0])
        dy = int((dy - eps) * self.current_histogram.shape[1])

        # Clamping dx and dy to the valid index range
        dx = max(0, min(dx, self.current_histogram.shape[0] - 1))
        dy = max(0, min(dy, self.current_histogram.shape[1] - 1))

        self.current_histogram[dx, dy] += 1

    def _key_to_action(self, key):
        if hasattr(self.action_space, "key_to_action"):
            return self.action_space.key_to_action(key)
        else:
            return key_to_action_default(key)

    def _keyboard_on_press(self, key):
        from pynput.keyboard import Key

        if key == Key.esc:
            self._terminate = True
            return False

        action = self._key_to_action(key)
        if action is not None:
            if action not in self._current_actions:
                self._current_actions.append(action)

    def _keyboard_on_release(self, key):
        action = self._key_to_action(key)
        if action is not None:
            if action in self._current_actions:
                self._current_actions.remove(action)

    # noinspection PyProtectedMember
    @staticmethod
    def play_human_mode(env, skip_frames=1, num_episodes=3, num_actions=None):
        from pynput.keyboard import Listener

        doom = env.unwrapped
        doom.skip_frames = 1  # handled by this script separately

        # noinspection PyProtectedMember
        def start_listener():
            with Listener(on_press=doom._keyboard_on_press, on_release=doom._keyboard_on_release) as listener:
                listener.join()

        listener_thread = Thread(target=start_listener)
        listener_thread.start()

        for episode in range(num_episodes):
            doom.mode = "human"
            env.reset()
            last_render_time = time.time()
            time_between_frames = 1.0 / 35.0

            total_rew = 0.0

            while not doom.game.is_episode_finished() and not doom._terminate:
                num_actions = 14 if num_actions is None else num_actions
                turn_delta_action_idx = num_actions - 1

                actions = [0] * num_actions
                for action in doom._current_actions:
                    if isinstance(action, int):
                        actions[action] = 1  # 1 for buttons currently pressed, 0 otherwise
                    else:
                        if action == "turn_left":
                            actions[turn_delta_action_idx] = -doom.delta_actions_scaling_factor
                        elif action == "turn_right":
                            actions[turn_delta_action_idx] = doom.delta_actions_scaling_factor

                for frame in range(skip_frames):
                    doom._actions_flattened = actions
                    _, rew, _, _, _ = env.step(actions)

                    new_total_rew = total_rew + rew
                    if new_total_rew != total_rew:
                        log.info("Reward: %.3f, total: %.3f", rew, new_total_rew)
                    total_rew = new_total_rew
                    state = doom.game.get_state()

                    verbose = True
                    if state is not None and verbose:
                        info = doom.get_info()
                        print(
                            "Health:",
                            info["HEALTH"],
                            # 'Weapon:', info['SELECTED_WEAPON'],
                            # 'ready:', info['ATTACK_READY'],
                            # 'ammo:', info['SELECTED_WEAPON_AMMO'],
                            # 'pc:', info['PLAYER_COUNT'],
                            # 'dmg:', info['DAMAGECOUNT'],
                        )

                    time_since_last_render = time.time() - last_render_time
                    time_wait = time_between_frames - time_since_last_render

                    if doom.show_automap and state.automap_buffer is not None:
                        map_ = state.automap_buffer
                        map_ = np.swapaxes(map_, 0, 2)
                        map_ = np.swapaxes(map_, 0, 1)
                        cv2.imshow("ViZDoom Automap Buffer", map_)
                        if time_wait > 0:
                            cv2.waitKey(int(time_wait) * 1000)
                    else:
                        if time_wait > 0:
                            time.sleep(time_wait)

                    last_render_time = time.time()

            if doom.show_automap:
                cv2.destroyAllWindows()

        log.debug("Press ESC to exit...")
        listener_thread.join()

    # noinspection PyProtectedMember
    @staticmethod
    def replay(env, rec_path):
        doom = env.unwrapped
        doom.mode = "replay"
        doom._ensure_initialized()
        doom.game.replay_episode(rec_path)

        episode_reward = 0
        start = time.time()

        while not doom.game.is_episode_finished():
            doom.game.advance_action()
            r = doom.game.get_last_reward()
            episode_reward += r
            log.info("Episode reward: %.3f, time so far: %.1f s", episode_reward, time.time() - start)

        log.info("Finishing replay")
        doom.close()
