---
title: "Pokemon Showdown RL Bot"
type: work
date: 2024-12-07T23:40:49+00:00
description: "Pokémon Showdown RL Bot"
caption: "This project focuses on using different Deep Reinforcement Learning methods to make a bot for playing the game Pokémon Showdown."
image: images/work/pokemon_showdown.jpeg
author: Barath Chandran, Pranjal Gautam
tags: ["Reinforcement Learning", "PyTorch"]
submitDate: June 27, 2024
github: https://github.com/dsgiitr/PikaBot-RL
---

## DRL Agents for Pokémon Showdown

Implementation of semi-Rainbow DQN, PPO, and REINFORCE to play the game Pokémon Showdown using PyTorch.

### Project Description
The action space in our implementation has 13 possible actions:
- **4 Moves**: Choose one of the four available moves for the current Pokémon.
- **5 Switches**: Switch to one of the five other Pokémon in your team.
- **4 Terastalized Moves**: Choose one of the four terastalized moves.

The state vector has a size of 54, which includes:
- **Expected Damage**: The expected damage of each move of the current Pokémon on each of the opposing Pokémon.
- **HP**: The hit points (HP) of all Pokémon displayed as a fraction.

We used `poke-env` to locally host a Pokémon Showdown server and connect the RL agents to it. The expected damages are calculated at each time step by integrating the Smogon damage calculator into `poke-env`.

We implemented the following three methods:
1. **Semi-Rainbow DQN**: A simpler variant of the Rainbow DQN algorithm combining several DQN improvements.
2. **PPO (Proximal Policy Optimization)**: A modern version of policy gradient method on an actor-critic network.
3. **REINFORCE**: A simpler monte-carlo version of policy gradient method.
