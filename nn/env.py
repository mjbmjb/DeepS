#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 06:52:02 2017

@author: mjb
"""
import torch
import Settings.game_settings as game_settings
import Game.card_to_string as card_to_string

from Game.card_tools import card_tools
card_tools = card_tools()

import Settings.arguments as arguments
import Settings.constants as constants
from Game.bet_sizing import BetSizing
from Game.Evaluation.evaluator import Evaluator
from Tree.tree_builder import PokerTreeBuilder
from Tree.strategy_filling import StrategyFilling

import random_card_generator
from nn.state import GameState

import random

filling = StrategyFilling()
builder = PokerTreeBuilder()
al_nn = []
rl_nn = []
evaluator = Evaluator()

class Env:

    def __init__(self):
        params = {}
        params['root_node'] = {}
        params['root_node']['board'] = card_to_string.string_to_board('')
        params['root_node']['street'] = 1
        params['root_node']['current_player'] = constants.players.P1
        params['root_node']['bets'] = arguments.Tensor([300, 300])
        params['limit_to_street'] = False
        builder = PokerTreeBuilder()
        self.root_node = builder.build_tree(params)
        filling.fill_uniform(self.root_node)
        self.state = GameState()
        self._cached_terminal_equities = {}
        
    def reset(self):
        self.state = GameState()
        pri_card = random_card_generator.generate_cards(constants.private_count * 2)
        self.state.private.append(pri_card[0:constants.private_count])
        self.state.private.append(pri_card[constants.private_count:])
        self.state.node = self.root_node
    
    #@return next_node, reward, terminal
    def step(self, state, action):
        parent_node = state.node
        if parent_node.terminal:
            # TODO ternimal value
#            terminal_value = 
            
            return None, parent_node.bets[parent_node.current_player] + 99, True
        # TODO grasp if action if invaild
        if action >= len(state.node.children):
            action = len(state.node.children) - 1
        
        assert (action < 4)
        next_node = state.node.children[action]
        if next_node.current_player == constants.players.chance:
            rannum = random.random()
            hand_id = int(state.private[parent_node.current_player][0])
            chance_strategy = parent_node.strategy[:,hand_id]
            for i in range(len(chance_strategy)):
                if rannum <= sum(chance_strategy[0: i+1]):
                    next_node = parent_node.children[i]
                    break
        
    #    next_state reward

       
        next_state = GameState()
        next_state.node = next_node
        next_state.private = state.private
        reward = next_node.bets[parent_node.current_player] - parent_node.bets[parent_node.current_player]
        terminal = False            

        return next_state, reward, terminal
                           
        #[0,1,1,1] means the second action
        
#    def _get_ternimal_equity(self, node):
#        if len(self._cached_terminal_equities == 0):
#            cached = TerminalEquity()
#            cached.set_board(node.board)
#            self._cached_terminal_equities[node.board_string] = cached
#        cached = self._cached_terminal_equities[node.board_string]
#        return cached


    def _get_terminal_value(self, state):
        node = state.node
        assert(node.ternimal)
        value = arguments.Tensor(2).fill_(-1)
        value[node.current_player] = 1
        if node.node_typee == constants.node_types.terminal_fold:
            #ternimal fold
            value.mul(node.bets[1 - node.current_player])
        else:
            # show down
            player_hand = self.private[node.current_player].tolist() + node.board.tolist()
            player_strength = evaluator.evaluate(player_hand, -1)
            oppo_hand = self.private[1 - node.current_player].tolist() + node.board.tolist()
            oppo_strength = evaluator.evaluate(oppo_hand, -1)
            
            if player_strength > oppo_strength:
                value.mul(node.bets[1 - node.current_player])
            else:
                value.mul(-node.bets[1 - node.current_player])
            return value
    def _al_action(self, state):
        
        # get possible bets in the node
        possible_bets = get_possible_actions()
        actions_count = possible_bets.size(0)
        
        # get the strategy
        
        
        assert(math.abs(1 - hand_strategy.sum()) < 0.001)
        # sample the action 
        action = strategy.cumsum(0).gt(random.random())
        
        return action
    
        
        