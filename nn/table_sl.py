#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 19:38:56 2017

@author: mjb
"""

import torch
import Settings.arguments as arguments
import Settings.constants as constants
import Settings.game_settings as game_settings

class TableSL:
    def __init__(self):
        # TODO count the num of node 652
        self.s_a_table = arguments.Tensor(34, game_settings.card_count * game_settings.private_count,\
                                     game_settings.actions_count).fill_(10)
        self.s_a_table[:,:,0:2].fill_(1)
    
    
    # @return action LongTensor[[]]
    def select_action(self, state):
        #TODO count the num of node
        #node id start from 1
        state_id = int(state.node.node_id - 1)
        hand_id = int(state.private[state.node.current_player][0])
        policy = self.s_a_table[state_id,hand_id,:] / self.s_a_table[state_id,hand_id,:].sum()

        random_num = torch.rand(1)
        for i in range(game_settings.actions_count):
            if random_num.sub_(policy[i])[0] <= 0:
                return arguments.LongTensor([[i]])


    def store(self, state, action):
        #node id start from 1
        state_id = int(state.node.node_id - 1)
        hand_id = int(state.private[state.node.current_player][0])
#        print('state node id')
#        print(state.node.node_id)
#        print('state id')
#        print(state_id)
#        print('self')
        action_id = action[0][0]
        self.s_a_table[state_id][hand_id][action_id] = self.s_a_table[state_id][hand_id][action_id]  + 1
