# Code for training and testing with REINFORCE in Pokémon Showdown

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
from torch.distributions import Categorical
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

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,action_std_init):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):

        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def select_action(self, state):
            with torch.no_grad():
                state = state.float()
                action, action_logprob, state_val = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

        # calculate advantages
        if old_states.dim() == 1:
            old_states = old_states.unsqueeze(1)
        if old_actions.dim() == 1:
            old_actions = old_actions.unsqueeze(1)
        if old_logprobs.dim() == 1:
            old_logprobs = old_logprobs.unsqueeze(1)
        if old_state_values.dim() == 1:
            old_state_values = old_state_values.unsqueeze(1)

        # Calculate advantages
        advantages = rewards - old_state_values.squeeze()

        # Ensure the advantages tensor has the correct shape
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       
class PPO_Player(Player):
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
        if self.state is not None:
            # observe R, S'
            self.reward = self.calc_reward(battle)
            next_state = self.embed_battle(battle)
            self.state = next_state
        else:
            # S first initialization
            self.state = self.embed_battle(battle)
        self.action = self.agent.select_action(torch.from_numpy(np.array(self.state)))
        self.agent.buffer.rewards.append(self.reward)
        self.agent.buffer.is_terminals.append(True if self.state is None else False)
        
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
        print(1 if self.current_battle.won else 0,end="")
        self.num_battles += 1
        self.agent.update()
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
        PPOag = PPO(state_dim=12, action_dim=9, lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, K_epochs=30, eps_clip=0.2)
        opponent = MaxDamagePlayer(battle_format="gen8ou", team=OUR_TEAM)
        player = PPO_Player(opponent=opponent,battle_format="gen8ou", team=OP_TEAM,agent=PPOag)
        await opponent.battle_against(opponent=player, n_battles=10000)
    
    asyncio.run(do_battle_training())
    
    