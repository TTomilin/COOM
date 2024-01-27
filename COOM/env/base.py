import gymnasium
import numpy as np
from typing import Dict, Any, Tuple, Optional


class BaseEnv(gymnasium.Env):
    """
    BaseEnv is an abstract base class for creating custom environments compatible with Gymnasium.

    This class serves as a template for defining environments for reinforcement learning tasks.
    It extends Gymnasium's Env class and provides a structured format for defining essential
    environment functionalities and properties.
    """

    def step(self, action):
        """
        Advances the environment by one step based on the given action.

        Args:
            action: The action to be taken in the environment. The type and format depend on the action space.

        Returns:
            A tuple (observation, reward, done, info):
                observation (np.ndarray): The current state observation of the environment.
                reward (float): The reward achieved by the previous action.
                done (bool): Whether the episode has ended.
                info (Dict[str, Any]): Additional information about the episode, environment, etc.
        """
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, ) -> Tuple[
        np.ndarray, Dict[str, Any]]:
        """
        Resets the environment to an initial state and returns the initial observation.

        Args:
            seed (Optional[int]): An optional random seed to ensure reproducibility.
            options (Optional[dict]): Additional configuration options for environment reset.

        Returns:
            A tuple (observation, info):
                observation (np.ndarray): The initial state observation of the environment.
                info (Dict[str, Any]): Additional information about the initial state.
        """
        raise NotImplementedError

    def render(self, mode="human"):
        """
        Renders the environment.

        The rendering can be to the screen, to an array, or other formats, depending on the mode.

        Args:
            mode (str): The mode to render with. Common modes include 'human' for screen display,
                        and 'rgb_array' for rendering to an array.
        """
        raise NotImplementedError

    @property
    def task(self) -> str:
        """
        Returns the current task of the environment.

        The task represents the current objective or scenario within the environment.

        Returns:
            str: The name or identifier of the current task.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Returns the name of the environment.

        This name typically represents the environment's scenario or task.

        Returns:
            str: The name of the environment.
        """
        raise NotImplementedError

    @property
    def task_id(self) -> int:
        """
        Returns the unique identifier for the current task.

        This ID is useful for distinguishing between different tasks within the sequence.

        Returns:
            int: The task's unique identifier.
        """
        raise NotImplementedError

    @property
    def num_tasks(self) -> int:
        """
        Returns the total number of tasks available in the environment.

        This is relevant for the continual learning environment with a sequence of tasks.

        Returns:
            int: The total number of tasks in the environment.
        """
        raise NotImplementedError

    @property
    def action_space(self) -> gymnasium.spaces.Discrete:
        """
        Defines the action space of the environment.

        This property should return a Discrete Gymnasium space object that specifies the format and
        constraints of valid actions in the environment.

        Returns:
            gymnasium.spaces.Space: The action space of the environment.
        """
        raise NotImplementedError

    @property
    def observation_space(self) -> gymnasium.Space:
        """
        Defines the observation space of the environment.

        This property should return a Gymnasium space object that specifies the format and constraints
        of valid observations in the environment.

        Returns:
            gymnasium.spaces.Space: The observation space of the environment.
        """
        raise NotImplementedError

    def get_statistics(self, mode: str = '') -> Dict[str, float]:
        """
        Retrieves statistics relevant to the current episode or the environment overall.

        Args:
            mode (str): An optional specifier to determine the type or category of statistics to retrieve.

        Returns:
            Dict[str, float]: A dictionary of statistical metrics.
        """
        raise NotImplementedError

    def clear_episode_statistics(self) -> None:
        """
        Clears or resets any statistics accumulated during an episode.

        This method is typically called at the end of an episode or during a reset.
        """
        raise NotImplementedError

    def get_active_env(self):
        """
        Returns the currently active sub-environment, if applicable.

        In scenarios where an environment comprises multiple sub-environments (e.g., in a continual learning setup),
        this method returns the currently active one.

        Returns:
            The active sub-environment.
        """
        raise NotImplementedError
