#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:52:24 2017

@author: mjb
"""

import sys
sys.path.append('/home/mjb/Nutstore/deepStack/')
import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string
import Settings.arguments as arguments
from Tree.tree_builder import PokerTreeBuilder
from Tree.tree_visualiser import TreeVisualiser
from Tree.tree_values import TreeValues
import Settings.constants as constants


def dfs_fill_table(node, table, builder):
    if node.terminal:
        return
    if node.current_player == constants.players.chance:
        node.table = arguments.Tensor([])
        node.rl = arguments.Tensor([])
        children = node.children
        for child in children:
            dfs_fill_table(child,table, builder)
        return
            
    # sl
    all_table = table[node.node_id,:,0:len(node.children)]
#    print(node.node_id)
    for i in range(all_table.size(0)):
        all_table[i,:] = all_table[i,:] / all_table[i,:].sum()
    
    node.strategy = torch.transpose(all_table,0,1).clone().fill_(1.0 / len(node.children))

    
    children = node.children
    for child in children:
        dfs_fill_table(child,table, builder)

model_num = 10000

builder = PokerTreeBuilder()

params = {}

params['root_node'] = {}
params['root_node']['board'] = card_to_string.string_to_board('As')
params['root_node']['street'] = 1
params['root_node']['current_player'] = constants.players.P1
params['root_node']['bets'] = arguments.Tensor([100, 100])
params['limit_to_street'] = False

tree = builder.build_tree(params)

table_sl = torch.load('/home/mjb/Nutstore/deepStack/Data/Model/Iter:' + str(model_num) + '.sl')

dfs_fill_table(tree, table_sl,builder)

#constract the starting range

start_range = arguments.Tensor(2, game_settings.card_count).fill_(1.0 / (game_settings.card_count - 1))
start_range[:,int(params['root_node']['board'][0])].fill_(0)

tree_values = TreeValues()
tree_values.compute_values(tree, start_range)



print('Exploitability: ' + str(tree.exploitability) + '[chips]' )

visualiser = TreeVisualiser()
visualiser.graphviz(tree,'test_values')