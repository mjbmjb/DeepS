#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 20:31:09 2017

@author: mjb
"""
from nn.table_sl import TableSL
from nn.env import Env

env = Env()
env.reset()
env.reset()
env.reset()
state = env.state
table_sl = TableSL()
for i in range(3):
    action = table_sl.select_action(state)
    print(action)
    table_sl.store(state, action)