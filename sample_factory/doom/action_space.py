import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

from sample_factory.algo.utils.spaces.discretized import Discretized


def doom_turn_attack():
    """
    TURN_LEFT
    TURN_RIGHT
    ATTACK
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
        )
    )  # noop, turn left, turn right  # noop, attack

    return space


def doom_turn_move():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
        )
    )  # noop, turn left, turn right  # noop, move forward

    return space


def doom_turn_move_use_jump():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    USE
    JUMP
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
            Discrete(2),
            Discrete(2),
        )
    )  # noop, turn left, turn right  # noop, move forward  # noop, use  # noop, jump

    return space


def doom_turn_move_use_jump_speed():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    USE
    JUMP
    SPEED
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
            Discrete(2),
            Discrete(2),
            Discrete(2),
        )
    )  # noop, turn left, turn right  # noop, move forward  # noop, use  # noop, jump  # noop, speed

    return space


def doom_turn_move_look_jump():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    MOVE_BACKWARD
    LOOK_UP
    LOOK_DOWN
    JUMP
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(3),
            Discrete(3),
            Discrete(2),
        )
    )  # noop, turn left, turn right  # noop, move forward, move backward  # noop, look up, look down  # noop, jump

    return space


def doom_turn_move_jump_accelerate():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    JUMP
    SPEED
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
            Discrete(2),
            Discrete(2),
        )
    )

    return space


def doom_turn_move_jump_accelerate_attack():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    JUMP
    SPEED
    ATTACK
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
            Discrete(2),
            Discrete(2),
            Discrete(2),
        )
    )

    return space


def doom_action_space():
    """
    Standard action space for full-featured Doom environments (e.g. deathmatch).
    This should precisely correspond to the available_buttons configuration in the .cfg file.
    This function assumes:
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        SELECT_NEXT_WEAPON
        SELECT_PREV_WEAPON
        ATTACK
        SPEED
        JUMP
        USE
        CROUCH
        TURN180
        LOOK_UP_DOWN_DELTA
        TURN_LEFT_RIGHT_DELTA
    """
    return gym.spaces.Tuple(
        (
            Discrete(3),  # noop, forward, backward
            Discrete(3),  # noop, move right, move left
            Discrete(3),  # noop, prev_weapon, next_weapon
            Discrete(2),  # noop, attack
            Discrete(2),  # noop, sprint
            Discrete(2),  # noop, jump
            Discrete(2),  # noop, use
            Discrete(2),  # noop, crouch
            Discrete(2),  # noop, turn180
            Box(np.float32(-1.0), np.float32(1.0), (1,)),  # [look_up, look_down]
            Box(np.float32(-1.0), np.float32(1.0), (1,)),  # [turn_left, turn_right]
        )
    )


def doom_action_space_no_speed():
    """
    Standard action space for full-featured Doom environments (e.g. deathmatch).
    This should precisely correspond to the available_buttons configuration in the .cfg file.
    This function assumes:
        MOVE_FORWARD
        MOVE_BACKWARD
        MOVE_RIGHT
        MOVE_LEFT
        SELECT_NEXT_WEAPON
        SELECT_PREV_WEAPON
        ATTACK
        JUMP
        USE
        CROUCH
        TURN180
        LOOK_UP_DOWN_DELTA
        TURN_LEFT_RIGHT_DELTA
    """
    return gym.spaces.Tuple(
        (
            Discrete(3),  # noop, forward, backward
            Discrete(3),  # noop, move right, move left
            Discrete(3),  # noop, prev_weapon, next_weapon
            Discrete(2),  # noop, attack
            Discrete(2),  # noop, jump
            Discrete(2),  # noop, use
            Discrete(2),  # noop, crouch
            Discrete(2),  # noop, turn180
            Box(np.float32(-1.0), np.float32(1.0), (1,)),  # [look_up, look_down]
            Box(np.float32(-1.0), np.float32(1.0), (1,)),  # [turn_left, turn_right]
        )
    )


def doom_action_space_no_move():
    """
    Standard action space for full-featured Doom environments (e.g. deathmatch).
    This should precisely correspond to the available_buttons configuration in the .cfg file.
    This function assumes:
        SELECT_NEXT_WEAPON
        SELECT_PREV_WEAPON
        ATTACK
        SPEED
        JUMP
        USE
        CROUCH
        TURN180
        LOOK_UP_DOWN_DELTA
        TURN_LEFT_RIGHT_DELTA
    """
    return gym.spaces.Tuple(
        (
            Discrete(3),  # noop, prev_weapon, next_weapon
            Discrete(2),  # noop, attack
            Discrete(2),  # noop, sprint
            Discrete(2),  # noop, jump
            Discrete(2),  # noop, use
            Discrete(2),  # noop, crouch
            Discrete(2),  # noop, turn180
            Box(np.float32(-1.0), np.float32(1.0), (1,)),  # [look_up, look_down]
            Box(np.float32(-1.0), np.float32(1.0), (1,)),  # [turn_left, turn_right]
        )
    )
