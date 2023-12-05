# Atari_RL

This repository contains implementation of 3 different agents DQN, Double-DQN, Dual-DQN for Atari-games supported by <a href=https://gymnasium.farama.org/>gym</a> library.
# Usage
* clone this repository into your local machine
<pre>
<code>
git clone https://github.com/Saran9999/Atari_RL
</code>
</pre>
* Usage in Python notebook
1) The code installs all the required libraries and APIs. You do not need to install any external libraries. Supported python and pip suffices.
2) Consider checking the references for better understanding.
   
* Usage in python 3.10
<pre>
<code>
pip install torchvision
pip install -q swig
pip3 install gym[all]
pip3 install gym[accept-rom-license]
</code>
</pre>

# How to Run:
* Run in Python notebook
Open the notebook in google collaboratory and select restart and run all. Training would take about 60 min for each one of the notebook.
* Running in Python 3.10
<pre>
<code>
python3 filename
</code>
</pre>
> Rendering requires GPU so please make sure to update your <B> GPU drivers</B>

# Code Walkthrough:
Each notebook file contains the following 3 classes:

## DQN
This contains implementation of the neural network that the agent uses to train upon the game and to select action for each query.
```python
def __init__(self, channels, num_actions):
         def __init__(self, channels, num_actions):
         '''
         This function defines the NN model we are going to estimate the value function
         '''  
         def forward(self, x):
         '''
         Forward function for the DQN network
         '''         
```

## ReplayBuffer
Contains implementation of the Replay Buffer to store experiences to sample them randomly while training.
```python
def __init__(self, buffer_size, batch_size):
       '''
        Initialize the Replay Buffer
       '''
    def add(self, state, action, reward, next_state, done):
       '''
        Add the experience into the Replay Buffer
       '''
    def sample(self):
        '''
        Sample batch of experiences from Replay Buffer
       '''
    def __len__(self):
        '''
        Return the size of Replay Buffer
       '''
```

## AgentDQN
This is our agent that makes the decision of actions for each timestep.
```python
class AgentDQN():
    def __init__(self, env):
       '''
       Intialize DQN Agent
       '''

    def save(self, save_path):
       '''
       Save the current model into save_path
       '''

    def load(self, load_path):
       '''
       Load model from the load_path
       '''

    def epsilon(self, step):
       '''
       Returns the Value of epsilon at current timestep
       '''

    def make_action(self, state, test=False):
       '''
       Returns the action predicted for both train and test
       '''

    def update(self):
       '''
       Update the Network after current batch is finished training
       '''

    def train(self):
        '''
        Main code for Agent Train
        '''
```
### Hyperparameters

1) The agent trains for 1,000,000 timesteps
2) Uses ReplayBuffer of size 100,000
3) Assumes that the discounted reward factor GAMMA = 0.99 and that each sequence of states comes to termination in finite no.of steps in future.
4) Uses RMSprop optimizer for weights in DQN neural network
5) Randomly selects actions for the initial 10,000 timesteps to fill up the ReplayBuffer to atleast a baseline.
* Try to change them as you please
```python
self.GAMMA = 0.99
# training hyperparameters
self.train_freq = 4 # frequency to train the online network
self.num_timesteps = 1000000# total training steps
self.learning_start = 10000 # before we start to update our network, we wait a few steps first to fill the replay.
self.batch_size = 32
self.display_freq = 25 # frequency to display training progress
self.target_update_freq = 1000 # frequency to update target network
# optimizer
self.optimizer = optim.RMSprop(self.online_net.parameters(), lr=1e-4)
self.steps = 0 # num. of passed steps. this may be useful in controlling exploration
self.eps_min = 0.025
self.eps_max = 1.0
self.eps_step = 60000
self.plot = {'steps':[], 'reward':[]}
# Initialize your replay buffer
self.memory = ReplayBuffer(100000, self.batch_size)
```

# REFS:
  https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-3-q-learning-with-neural-networks-algorithm-dqn-1e22ee928ecd
  
  https://markelsanz14.medium.com/introduction-to-reinforcement-learning-part-4-double-dqn-and-dueling-dqn-b349c9a61ea1
  
  https://colab.research.google.com/drive/16RjttswTuAjgqVV2jA-ioY2xVEQqIcQE#sandboxMode=true&scrollTo=OvvBAoQVJsuU
  
  https://arxiv.org/pdf/1312.5602v1.pdf
  
