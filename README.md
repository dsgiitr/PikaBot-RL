**Pokemon** is one of the most famous media franchises, spanning across different mediums such as video games, anime, movies, and manga for over 25 years. The Pokemon video games are adventure-based, turn-based strategy games where battles take place between Pokemon trainers. In each turn, players choose between four moves or maneuver their Pokemon while keeping track of the most optimal moves and strategies to win. This formula has remained mostly unchanged since the beginning of the franchise.

The idea for this project was to build a bot that can learn to play Pokemon, specifically to battle other trainers. The bot would learn the different mechanics of the game, from choosing the optimal moves each turn to making long-term strategies to win matches.

The easiest platform to develop such a bot is **Pokemon Showdown**, an online platform that is lightweight, free to play, and very accessible for this purpose. Previous work has also been done on similar projects, specifically with the **Poke-env** environment, which provides easy access to all the data needed, eliminating much of the technical implementation required for a classic Pokemon game.

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
