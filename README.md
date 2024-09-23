**Pokemon** is one of the most famous media franchises, spanning across different mediums such as video games, anime, movies, and manga for over 25 years. The Pokemon video games are adventure-based, turn-based strategy games where battles take place between Pokemon trainers. In each turn, players choose between four moves or maneuver their Pokemon while keeping track of the most optimal moves and strategies to win. This formula has remained mostly unchanged since the beginning of the franchise.

The idea for this project was to build a bot that can learn to play Pokemon, specifically to battle other trainers. The bot would learn the different mechanics of the game, from choosing the optimal moves each turn to making long-term strategies to win matches.

The easiest platform to develop such a bot is **Pokemon Showdown**, an online platform that is lightweight, free to play, and very accessible for this purpose. Previous work has also been done on similar projects, specifically with the **Poke-env** environment, which provides easy access to all the data needed, eliminating much of the technical implementation required for a classic Pokemon game.

![Alt text](https://forums.pokecharms.com/files/maxresdefault-jpg.555465/)

## Motivations for the Project

The goal is to build a bot for the online game **Pokemon Showdown** using reinforcement learning methods such as:

1. DDQN
2. PPO
3. Reinforce

The bot would be hosted on the online **Pokemon Showdown** server, allowing players to battle against it with the help of **Poke-env**.

## Methods Used

### 1)Reinforce
Reinforce is a **policy gradient** method that directly optimizes the agent's policy through trial and error by adjusting action probabilities based on rewards. It relies solely on the return from the environment to update the policy, without the need for a value function. While simple, it can be slow and less stable due to high variance in the updates, especially in complex environments with delayed rewards.

### 2)Proximal Policy Optimization
Proximal Policy Optimization (PPO) is a **policy gradient** method that improves on REINFORCE by using a **clipped objective** to prevent large, destabilizing policy updates. Unlike REINFORCE, PPO often pairs with an **Actor-Critic** architecture, where the critic estimates the value function to stabilize learning. Its stability and efficiency make it a more robust choice, especially for continuous and large-scale tasks.

### 3)DDQN
Double Deep Q-Network (DDQN) is a **value-based** method that refines the original DQN by separating action selection and evaluation to avoid overestimating Q-values. Unlike PPO and REINFORCE, which focus on learning a policy, DDQN learns the value of state-action pairs and uses these values to guide decision-making. This method is particularly effective in environments where learning precise action values is crucial for long-term success.



## Installation Instructions

1. **Ensure Python 3.8 or later and torch is installed** on your system.

2. **Install the required Python dependencies** using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Code for battle

To battle the bot, follow these steps:

1. **Create Two Pokémon Showdown Accounts**:
   - You need two accounts: one to host the bot and another for yourself.
   - Create these accounts at [Pokémon Showdown](https://play.pokemonshowdown.com).

2. **Prepare the Account Information**:
   - Create a file named `accounts.txt` in the same directory as your `PPO2.py` script.
   - This file should contain the username and password of the account you will use to host the bot.

3. **Run the PPO Script**:
   - Ensure the `PPO2.py` script, model weights file, and `accounts.txt` are all in the same folder.
   - Execute the script with the following command:

     ```bash
     python PPO2.py
     ```

4. **Set Up Your Team**:
   - Go to the Pokémon Showdown team builder and create a team using the following string. Copy and paste this string into the team builder:

     ```
     Qwilfish (Qwilfish-Hisui) @ Eviolite  
     Ability: Intimidate  
     Level: 83  
     Tera Type: Flying  
     EVs: 85 HP / 85 Atk / 85 Def / 85 SpA / 85 SpD / 85 Spe  
     - Toxic Spikes  
     - Crunch  
     - Gunk Shot  
     - Spikes  

     Medicham @ Choice Band  
     Ability: Pure Power  
     Level: 86  
     Tera Type: Fighting  
     EVs: 85 HP / 85 Atk / 85 Def / 85 SpA / 85 SpD / 85 Spe  
     - Zen Headbutt  
     - Ice Punch  
     - Poison Jab  
     - Close Combat  

     Orthworm @ Chesto Berry  
     Ability: Earth Eater  
     Level: 88  
     Tera Type: Electric  
     EVs: 85 HP / 85 Atk / 85 Def / 85 SpA / 85 SpD / 85 Spe  
     - Body Press  
     - Coil  
     - Rest  
     - Iron Tail  

     Chandelure @ Choice Scarf  
     Ability: Flash Fire  
     Level: 83  
     Tera Type: Fire  
     EVs: 85 HP / 85 Def / 85 SpA / 85 SpD / 85 Spe  
     IVs: 0 Atk  
     - Trick  
     - Shadow Ball  
     - Energy Ball  
     - Fire Blast  

     Floatzel @ Leftovers  
     Ability: Water Veil  
     Level: 85  
     Tera Type: Dark  
     EVs: 85 HP / 85 Atk / 85 Def / 85 SpA / 85 SpD / 85 Spe  
     - Crunch  
     - Low Kick  
     - Wave Crash  
     - Bulk Up  

     Spiritomb @ Leftovers  
     Ability: Infiltrator  
     Level: 90  
     Tera Type: Dark  
     EVs: 85 HP / 85 Atk / 85 Def / 85 SpA / 85 SpD / 85 Spe  
     - Poltergeist  
     - Toxic  
     - Foul Play  
     - Sucker Punch  
     ```

5. **Challenge the Bot**:
   - In Pokémon Showdown, use the search feature to find the username associated with the bot.
   - Challenge the bot. It should automatically accept the challenge.

6. **Enjoy the Battle**:
   - Have fun battling the bot!

