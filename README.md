# Atari_RL

This repository contains implementation of 3 different agents DQN, Double-DQN, Dual-DQN for Atari-games supported by <a href=https://gymnasium.farama.org/>gym</a> library.

1) Consider using a virtual environment for usage.
2) The code installs all the required libraries and APIs. You do not need to install any external libraries. Supported python and pip suffices.
3) Consider checking the references for better understanding.

# How to Run:
Open the notebook in google collaboratory and select restart and run all. Training would take about 60 min.

# Code Walkthrough:
Each notebook file contains the following 3 classes:

## DQN
```rst
This contains implementation of the neural network that the agent uses to train upon the game and to select action for each query.
```

## ReplayBuffer
```rst
Contains implementation of the buffer that is used to store the query and response as a tuple from the neural network.
```

## AgentDQN
```rst
This is our agent that makes the decision of actions for each timestep.

Few points to note on how the agent works:
1) The agent trains for 1,000,000 timesteps
2) Uses ReplayBuffer of size 100,000
3) Assumes that the discounted reward factor GAMMA = 0.99 and that each sequence of states comes to termination in finite no.of steps in future.
4) Uses RMSprop optimizer for weights in DQN neural network
5) Randomly selects actions for the initial 10,000 timesteps to fill up the ReplayBuffer to atleast a baseline.
```

# REFS:
  https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd
  https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1
  https://colab.research.google.com/drive/16RjttswTuAjgqVV2jA-ioY2xVEQqIcQE#sandboxMode=true&scrollTo=OvvBAoQVJsuU
  https://arxiv.org/pdf/1312.5602v1.pdf
  
