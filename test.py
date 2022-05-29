# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
Created on Sun May 29 13:05:27 2022
"""

import copy
import collections as coll
import itertools as it
import functools as ft
import operator as op
import os
import re

import io
import pickle as pk


import numpy as np
import pandas as pd
import scipy as sp
import sklearn as sk

import torch
import torch.nn as net


# cf: https://github.com/pytorch/pytorch/issues/16797

class CPU_Unpickler(pk.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


if __name__ == "__main__":

    # generator = pk.load(open("generator.pkl", 'rb'))

    # dev = torch.device('cpu')
    # model = torch.load('generator.pkl', map_location=torch.device("cpu"))
    # model = torch.load('generator.pkl', map_location=lambda storage, loc: dev)

    # my_model = net.load_state_dict(torch.load('generator.pkl', map_location=torch.device('cpu')))

    generator = CPU_Unpickler(open("generator.pkl", 'rb')).load()

    discriminator = CPU_Unpickler(open("discriminator.pkl", 'rb')).load()
