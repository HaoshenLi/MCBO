import numpy as np
import torch.nn as nn
import torch
import os
from models.model import *


class config_MCBO:
    def __init__(self):
        self.save_dir = 'results/MCBO/lr:1e-6_layer_1_heads_1' 
        self.model = MCBO()
        self.lr = 1e-6
        self.batch_size = 16
        self.epoches = 100
        self.milestones = []
        self.gamma = 0.1
        self.freeze_layers = False
        self.train = 'data_split/train_info.csv'
        self.val = 'data_split/val_info.csv'
        self.test = 'data_split/test_info.csv'
        self.which_eye = 'union'

class config_MCM:
    def __init__(self):
        self.save_dir = 'results/MCM/lr:1e-6_layer_1_heads_1' 
        self.model = MCM()
        self.lr = 1e-6
        self.batch_size = 16
        self.epoches = 100
        self.milestones = []
        self.gamma = 0.1
        self.freeze_layers = False
        self.train = 'data_split/train_info.csv'
        self.val = 'data_split/val_info.csv'
        self.test = 'data_split/test_info.csv'
        self.which_eye = 'union'

class config_BFM:
    def __init__(self):
        self.save_dir = 'results/BFM/lr:1e-6_layer_1_heads_1' 
        self.model = BFM()
        self.lr = 1e-6
        self.batch_size = 16
        self.epoches = 100
        self.milestones = []
        self.gamma = 0.1
        self.freeze_layers = False
        self.train = 'data_split/train_info.csv'
        self.val = 'data_split/val_info.csv'
        self.test = 'data_split/test_info.csv'
        self.which_eye = 'union'