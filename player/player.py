#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 05:12:49 2017

@author: mjb
"""

import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 建立连接:

s.connect(('www.sina.com.cn', 80))

