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




import numpy as np
import torch
import asyncio
from gymnasium.spaces import Space, Box

from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player import Player
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import ForfeitBattleOrder
from poke_env.data import GenData
from poke_env import to_id_str

import torch
from torch.autograd import Variable
import torch.nn.utils as utils
from DDQNAgent import DDQNAgent
from distutils.util import strtobool

import pandas as pd
import time
import json
import os


from collections import defaultdict
from datetime import date
from itertools import product
import argparse


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if len(battle.available_moves) > 0:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        else:
            return self.choose_random_move(battle)


class DQN_RLPlayer(Player):
    def __init__(self, rl_agent, opponent, battle_format="gen8ou", team=None):
        super().__init__(battle_format=battle_format, team=team)
        self.rl_agent = rl_agent
        self.num_battles = 0
        self.reward = 0
        self.state = None
        self.action = None
        self.action_space = list(range(9))
        self._reward_buffer = {}
        self.current_battle = None
    
    def choose_move(self, battle):
        self.current_battle = battle
        print(f"Turn {self.current_battle.turn}")
        if self.state is not None:
            # print(f"state: {self.state}")
            self.action = self.rl_agent.act(self.state)
            self.reward = self.calc_reward(battle)
            next_state = self.embed_battle(battle)
            self.rl_agent.step(self.state, self.action, self.reward, next_state, self.current_battle.finished)
            self.state = next_state
        else:
            self.state = self.embed_battle(battle)
            # print(f"state: {self.state}")
            self.action = self.rl_agent.act(self.state)

        print(f"Action: {self.action}")
        # time.sleep(1)
        if self.action == -1:
            return ForfeitBattleOrder()
        elif self.action < 4 and self.action < len(self.current_battle.available_moves) and not self.current_battle.force_switch:
            return self.create_order(self.current_battle.available_moves[self.action])
        elif self.action - 4 >= 0 and self.action - 4 < len(self.current_battle.available_switches):
            print(f"Switching to {self.current_battle.available_switches}")
            print(f"Switching number {self.action - 4}")
            return self.create_order(self.current_battle.available_switches[self.action - 4])
        else:
            # Check if there are no available switches and choose a random move instead
            return self.choose_random_move(self.current_battle)
        

    def embed_battle(self, battle):
        moves_power = -np.ones(4)
        gen_data = GenData.from_gen(8)
        type_chart = gen_data.load_type_chart(8)

        for i, move in enumerate(self.current_battle.available_moves):
            # print(move)
            moves_power[i] = (
                move.base_power / 100 * move.type.damage_multiplier(
                    self.current_battle.opponent_active_pokemon.type_1,
                    self.current_battle.opponent_active_pokemon.type_2,
                    type_chart=type_chart
                )
            )

        my_health =  [self.current_battle.active_pokemon.current_hp_fraction] + [mon.current_hp_fraction for mon in self.current_battle.team.values() if mon != self.current_battle.active_pokemon]
        # print(f"my health{my_health}")
        opponent_health = self.current_battle.opponent_active_pokemon.current_hp_fraction
        final_vector = np.concatenate([moves_power, my_health, [opponent_health]])
        return np.float32(final_vector)

    def reward_computing_helper(self, battle, fainted_value=0.15, hp_value=0.15, number_of_pokemons=6, 
                                starting_value=0.0, status_value=0.15, victory_value=1.0):
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in self.current_battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(self.current_battle.team)) * hp_value

        for mon in self.current_battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(self.current_battle.opponent_team)) * hp_value

        if self.current_battle.won:
            current_value += victory_value
        elif self.current_battle.lost:
            current_value -= victory_value

        reward = current_value - self._reward_buffer[battle]
        self._reward_buffer[battle] = current_value

        return reward

    def calc_reward(self, battle):
        return self.reward_computing_helper(battle, fainted_value=2, hp_value=1, victory_value=15)


    def _battle_finished_callback(self, battle):
        self.num_battles += 1
        self.rl_agent.step(self.state, self.action, self.reward, self.state, True)

    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return Box(np.array(low, dtype=np.float32), np.array(high, dtype=np.float32), dtype=np.float32)




async def do_battle_validation():
    agent1 = DDQNAgent(state_size=11, action_size=9)
    opponent = MaxDamagePlayer(battle_format="gen8ou", team=OP_TEAM)
    player = DQN_RLPlayer(opponent=opponent, battle_format="gen8ou", team=OUR_TEAM, rl_agent=agent1)
    print("Challenges sent")
    # opponent.opponent = player
    await opponent.battle_against(player, n_battles=1000)
    # await player.send_challenges(
    #     opponent=to_id_str(opponent.username),
    #     n_challenges=1,
    # )
    # await opponent.send_challenges(opponent=to_id_str(player.username), n_challenges=1)

asyncio.run(do_battle_validation())

# import nest_asyncio
# nest_asyncio.apply()
# loop = asyncio.get_event_loop()
# loop.run_until_complete(loop.create_task(do_battle_validation()))







