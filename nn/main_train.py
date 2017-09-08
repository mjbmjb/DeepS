#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 05:15:26 2017

@author: mjb
"""

import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')

import random
import torch
import numpy as np
import Settings.arguments as arguments
import Settings.constants as constants
from itertools import count
from nn.env import Env
from nn.dqn import DQN
from nn.dqn import DQNOptim
from nn.table_sl import TableSL
from Tree.tree_builder import PokerTreeBuilder
builder = PokerTreeBuilder()

dqn_optim = DQNOptim()
table_sl = TableSL()
num_episodes = 10
env = Env()


def get_action(state, flag):
    # flag = 0 sl flag = 1 rl
    action = table_sl.select_action(state) if flag == 0 else dqn_optim.select_action(state)
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

time_start = 0
#@profile
def main():
    import time
    time_start = time.time()
    total_reward = 0.0
    
    for i_episode in range(arguments.epoch_count + 1):
        # choose policy 0-sl 1-rl
        flag = 0 if random.random() > arguments.eta else 1
        
        # Initialize the environment and state
        env.reset()
        state = env.state
        for t in count():
            state_tensor = builder.statenode_to_tensor(state)
            # Select and perform an action
            assert(state_tensor.size(1) == 20)
            
            if flag == 0:
                # sl
                action = table_sl.select_action(state)
            else:
                #rl
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
                table_sl.store(state, action)
                
    
            # Perform one step of the optimization (on the target network)
            dqn_optim.optimize_model() 
            # Move to the next state
            state = next_state
    
    
            #accumlate the reward
            total_reward = total_reward + reward
            
            if done:
                dqn_optim.episode_durations.append(t + 1)
#                dqn_optim.plot_durations()
                break
            
    # save the model
    path = '../Data/Model/'
    sl_name = path + "Iter:" + str(i_episode) + '.sl'
    rl_name = path + "Iter:" + str(i_episode) + '.rl'
    memory_name = path + 'Iter:' + str(i_episode)   
    # save sl strategy
    torch.save(table_sl.s_a_table, sl_name)
    # save rl strategy
    # 1.0 save the prarmeter
    torch.save(dqn_optim.model.state_dict(), rl_name)
    # 2.0 save the memory of DQN
    np.save(memory_name, np.array(dqn_optim.memory.memory))
            
            
    print('Complete')
    print((time.time() - time_start))
    print(total_reward / arguments.epoch_count)
#    dqn_optim.plt.ioff()
#    dqn_optim.plt.show()

if __name__ == '__main__':
    main()


            
