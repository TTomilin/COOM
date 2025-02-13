import gymnasium as gym
import numpy as np
from vizdoom import GameVariable

WEAPON_WEIGHTS = np.linspace(0.1, 1.0, 7)
DECOY_WEIGHTS = np.linspace(0.25, 1.0, 4)
CARRYING_CAPACITY = 1.0
MIN_SPEED = 0.25
HARD_CONSTRAINT_PENALTY = 10
POSTPONED_PENALTY_MULTIPLIER = 0.75


class ArmamentBurdenCostFunction(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.load = 0
        self.deaths = 0
        self.discards = 0
        self.deliveries = 0
        self.total_cost = 0
        self.delivery_cost = 0
        self.num_decoys_carried = 0
        self.num_weapons_carried = 0
        self.weapon_pickup_reward = 0
        self.total_reward_delivery = 0
        self.total_decoys_acquired = 0
        self.total_weapons_acquired = 0
        self.reward_current_delivery = 0
        self.hard_constraint = env.unwrapped.hard_constraint

    def reset(self, **kwargs):
        self.reset_delivery()
        self.deaths = 0
        self.discards = 0
        self.deliveries = 0
        self.total_cost = 0
        self.weapon_pickup_reward = 0
        self.total_reward_delivery = 0
        self.total_decoys_acquired = 0
        self.total_weapons_acquired = 0
        return self.env.reset(**kwargs)

    def reset_delivery(self):
        self.load = 0
        self.delivery_cost = 0
        self.num_decoys_carried = 0
        self.num_weapons_carried = 0
        self.reward_current_delivery = 0

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        weapon_id = int(self.game.get_game_variable(GameVariable.USER1))
        num_weapons_carried = int(self.game.get_game_variable(GameVariable.USER2))
        decoy_id = int(self.game.get_game_variable(GameVariable.USER5))
        num_decoys_carried = int(self.game.get_game_variable(GameVariable.USER6))

        health = self.game.get_game_variable(GameVariable.HEALTH)

        excess = 0.0
        player_speed = 1.0
        cost_this_step = 0.0
        self.weapon_pickup_reward += reward

        # Check if the player fell into a lava pit
        if health == 0:
            cost_this_step += 10
            self.deaths += 1

        # Determine whether the player has discarded the inventory
        discard = bool(self.game.get_game_variable(GameVariable.USER4))
        if discard:
            cost_this_step += 0.1
            self.discards += 1
            self.reset_delivery()

        # Check if a weapon or decoy has been picked up
        weapon_obtained = num_weapons_carried > self.num_weapons_carried
        decoy_obtained = num_decoys_carried > self.num_decoys_carried

        if weapon_obtained:
            weight = WEAPON_WEIGHTS[weapon_id]

            self.num_weapons_carried += 1  # Number of weapons carried for this delivery
            self.total_weapons_acquired += 1  # Number of weapons picked up throughout the entire episode

            # Calculate the new potential reward to see if it would exceed the carrying capacity
            self.reward_current_delivery += weight

            # Increase load with the weight of the new weapon
            self.load += weight

        if decoy_obtained:
            self.num_decoys_carried += 1  # Number of decoys carried for this delivery
            self.total_decoys_acquired += 1  # Number of decoys picked up throughout the entire episode

            # Increase load with the weight of the obtained decoy item
            self.load += DECOY_WEIGHTS[decoy_id]

        # Check if the total reward from weapons exceeds the carrying capacity and apply the soft constraint penalty
        if self.load > CARRYING_CAPACITY and not self.hard_constraint:
            # Calculate the excess amount over the capacity
            excess = self.load - CARRYING_CAPACITY

            # Calculate the excess percentage over the initial carrying capacity threshold
            excess_percentage = excess / CARRYING_CAPACITY

            # Calculate new speed
            player_speed -= excess_percentage
            player_speed = max(player_speed, MIN_SPEED)

            # The excess weight is proportional to the cost
            # Incur full cost if a weapon or decoy was obtained this step, otherwise apply a fraction
            cost_this_step = excess_percentage if weapon_obtained or decoy_obtained else POSTPONED_PENALTY_MULTIPLIER * excess_percentage
            self.delivery_cost += cost_this_step

        # Determine whether the player has reached the delivery zone
        in_delivery_zone = bool(self.game.get_game_variable(GameVariable.USER3))
        if in_delivery_zone:
            if self.num_weapons_carried > 0:
                self.deliveries += 1  # Increment the number of successful deliveries made
            self.total_reward_delivery += self.reward_current_delivery
            self.reset_delivery()

        # Carrying capacity breached for hard constraint
        if self.hard_constraint:
            if num_weapons_carried < self.num_weapons_carried:
                cost_this_step += HARD_CONSTRAINT_PENALTY
                self.delivery_cost += cost_this_step
                self.num_weapons_carried = 0
                self.num_decoys_carried = 0
            speed_reduction = bool(self.game.get_game_variable(GameVariable.USER5))
            if speed_reduction:
                cost_this_step += HARD_CONSTRAINT_PENALTY * POSTPONED_PENALTY_MULTIPLIER
                player_speed = 0.1

        self.total_cost += cost_this_step

        info['cost'] = cost_this_step
        info["true_objective"] = self.total_reward_delivery
        info["episode_extra_stats"] = {
            'deaths': self.deaths,
            'cost_this_step': cost_this_step,
            'cost': self.total_cost,
            'delivery_cost': self.delivery_cost,
            'total_cost': self.total_cost,
            'weapons_acquired': self.total_weapons_acquired,
            'decoys_acquired': self.total_decoys_acquired,
            'deliveries': self.deliveries,
            'player_speed': player_speed,
            'excess_weight': excess,
            'discards': self.discards,
            'weapon_pickup_reward': self.weapon_pickup_reward,
            'reward_delivery': self.total_reward_delivery
        }

        return observation, reward, terminated, truncated, info
