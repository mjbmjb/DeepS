#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 05:15:26 2017

@author: mjb
"""

import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')
#import cProfilev

import random
import torch
import numpy as np
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings
from itertools import count
from nn.env import Env
from nn.dqn import DQN
from nn.dqn import DQNOptim
from nn.table_sl import TableSL
from nn.state import GameState
from Tree.tree_builder import PokerTreeBuilder
from Tree.Tests.test_tree_values import ValuesTester
from collections import namedtuple
builder = PokerTreeBuilder()

num_episodes = 10
env = Env()
value_tester = ValuesTester()

Agent = namedtuple('Agent',['rl','sl'])

agent = Agent(rl=DQNOptim(),sl=TableSL())
dqn_optim = agent.rl
table_sl = agent.sl


def load_model(dqn_optim, iter_time):
    iter_str = str(iter_time)
    # load rl model (only the net)
    dqn_optim.model.load_state_dict(torch.load('../Data/Model/Iter:' + iter_str + '.rl'))
    # load sl model
    table_sl.s_a_table = torch.load('../Data/Model/Iter:' + iter_str + '.sl')

def save_model(episod):
    path = '../Data/Model/'
    sl_name = path + "Iter:" + str(episod) + '.sl'
    rl_name = path + "Iter:" + str(episod) + '.rl'
    memory_name = path + 'Iter:' + str(episod)   
    # save sl strategy
    torch.save(table_sl.s_a_table, sl_name)
    # save rl strategy
    # 1.0 save the prarmeter
    torch.save(dqn_optim.model.state_dict(), rl_name)
    # 2.0 save the memory of DQN
    np.save(memory_name, np.array(dqn_optim.memory.memory))

def save_table_csv(table):
    with open('../Data/table.csv', 'a') as fout:
        for i in range(table.size(0)):
            fout.write(str(table[i].sum()))
            fout.write(',')
        fout.write('\n')
    
    


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
    
    if arguments.load_model:
        load_model(dqn_optim, arguments.load_model_num)
    
    for i_episode in range(arguments.epoch_count + 1):
        # choose policy 0-sl 1-rl
        flag = 0 if random.random() > arguments.eta else 1
        
        # Initialize the environment and state
        env.reset()
        state = env.state
        for t in count():
            state_tensor = builder.statenode_to_tensor(state)
            # Select and perform an action
            assert(state_tensor.size(1) == 32)
            
            if flag == 0:
                # sl
                action = table_sl.select_action(state)
            else:
                #rl
                action = dqn_optim.select_action(state_tensor)
                
            next_state, real_next_state, reward, done = env.step(agent, state, int(action[0][0]))
            
            # transform to tensor
            real_next_state_tensor = builder.statenode_to_tensor(real_next_state)
            
            reward_tensor = arguments.Tensor([reward])
#            reward_tensor = arguments.Tensor([reward])
            action_tensor = action
            
            # Store the transition in reforcement learning memory Mrl
            dqn_optim.memory.push(state_tensor, action_tensor, real_next_state_tensor, reward_tensor)
            if flag == 1:
                # if choose sl store tuple(s,a) in supervised learning memory Msl
                table_sl.store(state, action)
                
    
            # Perform one step of the optimization (on the target network)
            dqn_optim.optimize_model() 
            # Move to the next state
            state = next_state
    
            # update the target net work
            if dqn_optim.steps_done > 0 and dqn_optim.steps_done % 300 == 0:
                dqn_optim.target_net.load_state_dict(dqn_optim.model.state_dict())
#                dqn_optim.plot_error_vis()
            
            
#            if i_episode % 100 == 0:
#                    dqn_optim.plot_error_vis(i_episode)
            
            if done:
                if(i_episode % 100 == 0):
                    dqn_optim.plot_error_vis(i_episode)
                if(i_episode % arguments.save_epoch == 0):
                    save_model(i_episode)
                    value_tester.test(table_sl.s_a_table.clone(), i_episode)
#                    save_table_csv(table_sl.s_a_table)
#                dqn_optim.episode_durations.append(t + 1)
#                dqn_optim.plot_durations()
                break
            
            
#    dqn_optim.plot_error()
#    global LOSS_ACC
#    LOSS_ACC = dqn_optim.error_acc
    # save the model
    if arguments.load_model:
        i_episode = i_episode + arguments.load_model_num
            
            
    print('Complete')
    print((time.time() - time_start))
#    dqn_optim.plt.ioff()
#    dqn_optim.plt.show()

if __name__ == '__main__':
#    cProfile.run(main())
    main()


            
