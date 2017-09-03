#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 06:41:46 2017

@author: mjb
"""

import Settings.game_settings as game_settings
import Settings.arguments as arguments
import Settings.constants as constants
from TerminalEquity.terminal_equity import TerminalEquity 
from Game.card_tools import card_tools
card_tools = card_tools()

class TreeEnv:
    def __init__(self):
        self._cached_terminal_equities = {}

    def _get_terminal_equity(self, node):
        if self._cached_terminal_equities.has_key(node.board_string):
            return self._cached_terminal_equities[node.board_string]
        else:
            cached = TerminalEquity()
            cached.set_board(node.board)
            self._cached_terminal_equities[node.board_string] = cached
        return cached
        
    