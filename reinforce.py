
import numpy as np
import torch
import asyncio

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
import gymnasium
from gymnasium.spaces import Space, Box

import sys
import math
import numpy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
import torchvision.transforms as T
from torch.autograd import Variable
import pdb
# Policy class
class Policy(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Policy, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.linear1(x))
        action_scores = self.linear2(x)
        return action_scores

# REINFORCE agent
class REINFORCE:
    def __init__(self, hidden_size, num_inputs, action_space, lr):
        self.action_space = action_space
        self.model = Policy(hidden_size, num_inputs, action_space).float()
        self.model = self.model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

    def select_action(self, state):
        state = torch.from_numpy(numpy.array(state)).float()
        probs = self.model(Variable(state).float())  
        #action = probs.multinomial(1).data
        action = torch.argmax(probs).data
        prob = probs[action].view(1, -1)
        log_prob = prob.log()
        entropy = - (probs*probs.log()).sum()

        return action, log_prob, entropy

    def update_parameters(self, rewards, log_probs, entropies, gamma):
        R = torch.zeros(1, 1)
        loss = 0
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R))).sum() - (0.0001*entropies[i]).sum()
        loss = loss / len(rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(self.model.parameters(), 40)
        self.optimizer.step()
# Code for training and testing with REINFORCE in Pokémon Showdown

import numpy as np
import torch
import asyncio

from poke_env.player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.player import Player
from poke_env.data import GenData

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
from distutils.util import strtobool

import pandas as pd
import time
import json
import os
from collections import defaultdict
from datetime import date
from itertools import product
import argparse

# Definition of the agent stochastic team (Pokémon Showdown template)
OUR_TEAM = """
Pikachu-Original (M) @ Light Ball  
Ability: Static  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Volt Tackle  
- Nuzzle  
- Iron Tail  
- Knock Off

Charizard @ Life Orb  
Ability: Solar Power  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Flamethrower  
- Dragon Pulse  
- Roost  
- Sunny Day

Blastoise @ White Herb  
Ability: Torrent  
EVs: 4 Atk / 252 SpA / 252 Spe  
Mild Nature  
- Scald  
- Ice Beam  
- Earthquake  
- Shell Smash

Venusaur @ Black Sludge  
Ability: Chlorophyll  
EVs: 252 SpA / 4 SpD / 252 Spe  
Modest Nature  
IVs: 0 Atk  
- Giga Drain  
- Sludge Bomb  
- Sleep Powder  
- Leech Seed

Sirfetch’d @ Aguav Berry  
Ability: Steadfast  
EVs: 248 HP / 252 Atk / 8 SpD  
Adamant Nature  
- Close Combat  
- Swords Dance  
- Poison Jab  
- Knock Off

Tauros (M) @ Assault Vest  
Ability: Intimidate  
EVs: 252 Atk / 4 SpD / 252 Spe  
Adamant Nature  
- Double-Edge  
- Earthquake  
- Megahorn  
- Iron Head  
"""

# Definition of the opponent stochastic team (Pokémon Showdown template)
OP_TEAM = """
Eevee @ Eviolite  
Ability: Adaptability  
EVs: 252 HP / 252 Atk / 4 SpD  
Jolly Nature  
- Quick Attack  
- Flail  
- Facade  
- Wish

Vaporeon @ Leftovers  
Ability: Hydration  
EVs: 252 HP / 252 Def / 4 SpA  
Bold Nature  
IVs: 0 Atk  
- Scald  
- Shadow Ball  
- Toxic  
- Wish

Sylveon @ Aguav Berry  
Ability: Pixilate  
EVs: 252 HP / 252 SpA / 4 SpD  
Modest Nature  
IVs: 0 Atk  
- Hyper Voice  
- Mystical Fire  
- Psyshock  
- Calm Mind

Jolteon @ Assault Vest  
Ability: Quick Feet  
EVs: 252 SpA / 4 SpD / 252 Spe  
Timid Nature  
IVs: 0 Atk  
- Thunderbolt  
- Hyper Voice  
- Volt Switch  
- Shadow Ball

Leafeon @ Life Orb  
Ability: Chlorophyll  
EVs: 252 Atk / 4 SpD / 252 Spe  
Jolly Nature  
- Leaf Blade  
- Knock Off  
- X-Scissor  
- Swords Dance

Umbreon @ Iapapa Berry  
Ability: Inner Focus  
EVs: 252 HP / 4 Atk / 252 SpD  
Careful Nature  
- Foul Play  
- Body Slam  
- Toxic  
- Wish  
"""

# Encoding stochastic Pokémon Name for ID
NAME_TO_ID_DICT = {
    "pikachuoriginal": 0,
    "charizard": 1,
    "blastoise": 2,
    "venusaur": 3,
    "sirfetchd": 4,
    "tauros": 5,
    "eevee": 0,
    "vaporeon": 1,
    "sylveon": 2,
    "jolteon": 3,
    "leafeon": 4,
    "umbreon": 5
}

np.random.seed(0)

# Definition of REINFORCE player
class ReinforcePlayer(Player):
    def __init__(self,opponent, battle_format, team,agent):
        super().__init__(battle_format=battle_format, team=team)
        self.agent = agent
        self.state = None
        self.action = None
        self.reward = 0
        self.num_battles = 0
        self.entropies = []
        self.log_probs = []
        self.rewards = []
        self._reward_buffer={}
        self.current_battle=None

    def choose_move(self, battle):
        self.current_battle = battle
        #print(f"Turn {self.current_battle.turn}")
        if self.state is not None:
            # observe R, S'
            self.reward = self.calc_reward(battle)
            next_state = self.embed_battle(battle)
            # S <- S'
            self.state = next_state
        else:
            # S first initialization
            self.state = self.embed_battle(battle)

        # choose A from S using policy pi
        self.action, entropy, log_prob = self.agent.select_action(torch.from_numpy(np.array(self.state)))

        self.entropies.append(entropy)
        self.log_probs.append(log_prob)
        self.rewards.append(torch.from_numpy(np.array(self.reward)))

        # if the selected action is not possible, perform a random move instead
        if self.action == -1:
            return ForfeitBattleOrder()
        elif self.action < 4 and self.action < len(self.current_battle.available_moves) and not self.current_battle.force_switch:
            return self.create_order(self.current_battle.available_moves[self.action])
        elif 0 <= self.action - 4 < len(self.current_battle.available_switches):
            return self.create_order(self.current_battle.available_switches[self.action - 4])
        else:
            return self.choose_random_move(self.current_battle)
    
    

    def embed_battle(self, battle):
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        gen_data = GenData.from_gen(8)
        type_chart = gen_data.load_type_chart(8)
        for i, move in enumerate(self.current_battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    self.current_battle.opponent_active_pokemon.type_1,
                    self.current_battle.opponent_active_pokemon.type_2,
                    type_chart=type_chart
                )

        # We count how many pokemons have not fainted in each team
        n_fainted_mon_team = (
            len([mon for mon in self.current_battle.team.values() if mon.fainted])
        )
        n_fainted_mon_opponent = (
            len([mon for mon in self.current_battle.opponent_team.values() if mon.fainted])
        )
        state = np.concatenate([
            [NAME_TO_ID_DICT[str(self.current_battle.active_pokemon).split(' ')[0]]],
            [NAME_TO_ID_DICT[str(self.current_battle.opponent_active_pokemon).split(' ')[0]]],
            [move_base_power for move_base_power in moves_base_power],
            [move_dmg_multiplier for move_dmg_multiplier in moves_dmg_multiplier],
            [n_fainted_mon_team,
             n_fainted_mon_opponent]])
        return state

    # Computing rewards
    def reward_computing_helper(
            self,
            battle: AbstractBattle,
            *,
            fainted_value: float = 0.15,
            hp_value: float = 0.15,
            number_of_pokemons: int = 6,
            starting_value: float = 0.0,
            status_value: float = 0.15,
            victory_value: float = 1.0
    ) -> float:
        # 1st compute
        if self.current_battle not in self._reward_buffer:
            self._reward_buffer[self.current_battle] = starting_value
        current_value = 0

        # Verify if pokemon have fainted or have status
        for mon in self.current_battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(self.current_battle.team)) * hp_value

        # Verify if opponent pokemon have fainted or have status
        for mon in self.current_battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(self.current_battle.opponent_team)) * hp_value

        # Verify if we won or lost
        if self.current_battle.won:
            current_value += victory_value
        elif self.current_battle.lost:
            current_value -= victory_value

        # Value to return
        to_return = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value
        return to_return

    # Calling reward_computing_helper
    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)

    def _battle_finished_callback(self, battle):
        self.num_battles += 1
        self.agent.update_parameters(self.rewards, self.log_probs, self.entropies, 0.95)
        print(1 if self.current_battle.won else 0,end="")
        self.entropies = []
        self.log_probs = []
        self.rewards = []

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)

if __name__ == "__main__":

    OUR_TEAM = OUR_TEAM
    OP_TEAM = OP_TEAM


     # training

    async def do_battle_training():
        reinforce_agent = REINFORCE(64, 12, 13, lr = 1e-3)
        opponent = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
        player = ReinforcePlayer(opponent=opponent,battle_format="gen8ou", team=OUR_TEAM, agent=reinforce_agent)
        await opponent.battle_against(opponent=player, n_battles=100)
        return reinforce_agent
    
    asyncio.run(do_battle_training())
    