#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 05:15:26 2017

@author: mjb
"""

import random
import Settings.arguments as arguments
import Settings.constants as constants
from itertools import count
from nn.env import Env
from nn.dqn import DQN
from nn.dqn import DQNOptim
from Tree.tree_builder import PokerTreeBuilder
builder = PokerTreeBuilder()

dqn_optim = DQNOptim()
num_episodes = 10
env = Env()


def get_action(state, flag):
    # flag = 0 sl flag = 1 rl
#    action = _al_action(state) if flag == 0 else dqn_optim.select_action(state)
    action = dqn_optim.select_action(state)
    return action


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.

for i_episode in range(arguments.epoch_count):
    # choose policy 0-sl 1-rl
    flag = 0 if random.random() > arguments.eta else 1
    
    # Initialize the environment and state
    env.reset()
    state = env.state
    for t in count():
        state_tensor = builder.statenode_to_tensor(state)
        # Select and perform an action
        assert(state_tensor.size(1) == 20) 
        action = dqn_optim.select_action(state_tensor)
        next_state, reward, done = env.step(state, int(action[0][0]))
        
        
        # transform to tensor
        next_state_tensor = builder.statenode_to_tensor(next_state)
        reward_tensor = arguments.Tensor([reward])
        action_tensor = action
        
        # Store the transition in reforcement learning memory Mrl
        dqn_optim.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)
        if flag == 1:
            # if choose sl store tuple(s,a) in supervised learning memory Msl
            pass

        # Perform one step of the optimization (on the target network)
        dqn_optim.optimize_model() 
        # Move to the next state
        state = next_state


        if done:
            dqn_optim.episode_durations.append(t + 1)
            dqn_optim.plot_durations()
            break

print('Complete')
dqn_optim.plt.ioff()
dqn_optim.plt.show()
