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
        # TODO count the num of node
        self.s_a_table = arguments.Tensor(382 * game_settings.card_count * game_settings.private_count,\
                                     game_settings.actions_count).fill_(1)
    
    
    # @return action LongTensor[[]]
    def select_action(self, state):
        #TODO count the num of node
        #node id start from 1
        state_id = int(((state.node.node_id - 1) * 8 + state.private[state.node.current_player])[0])
        policy = self.s_a_table[state_id,:] / self.s_a_table[state_id,:].sum()

        random_num = torch.rand(1)
        for i in range(game_settings.actions_count):
            if random_num.sub_(policy[i])[0] <= 0:
                return arguments.LongTensor([[i]])


    def store(self, state, action):
        #node id start from 1
        state_id = int(((state.node.node_id - 1) * 8 + state.private[state.node.current_player])[0])
#        print('state node id')
#        print(state.node.node_id)
#        print('state id')
#        print(state_id)
#        print('self')
        action_id = action[0][0]
        self.s_a_table[state_id][action_id] = self.s_a_table[state_id][action_id] + 1
