import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.decomposition import PCA
from PIL import Image
from torchvision import transforms
import cv2
import pdb


class MyDataSet(Dataset):

    def __init__(self, csv_path: str, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.left_path_list = pd.read_csv(self.csv_path)['left_eye'].to_list()
        self.right_path_list = pd.read_csv(self.csv_path)['right_eye'].to_list()
        self.label_list = pd.read_csv(self.csv_path)['label'].to_list()
        self.pd_data = pd.read_csv(self.csv_path)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        left_file = self.left_path_list[index]
        right_file = self.right_path_list[index]
        left_path = os.path.join(left_file, 'RGB.png')
        right_path = os.path.join(right_file, 'RGB.png')
        Limg = Image.open(left_path)
        Rimg = Image.open(right_path)
        label = self.label_list[index]
        label = torch.tensor(int(label))
        clicinfo = torch.LongTensor([
            self._info_sex(self.pd_data.iloc[index]["SEX"]),
            self._info_age(self.pd_data.iloc[index]["AGENEW2014"]),
            self._info_bmi(self.pd_data.iloc[index]["BMI"]),
            self._info_smoke(self.pd_data.iloc[index]["SMOKE_CUR"]),
            self._info_drink(self.pd_data.iloc[index]["DRINK_CUR"]),
        ])
        if self.transform is not None:
            Limg = self.transform(Limg)
            Rimg = self.transform(Rimg)
        return Limg, Rimg, label, clicinfo
    
    def _info_sex(self, x):
        if x == 1: # man
            return 1
        elif x == 2: # woman
            return 2
        return 0
    
    def _info_age(self, x):
        if x < 40: # young
            return 1
        elif x >= 40 and x < 60: # middle
            return 2
        elif x >= 60: # old
            return 3
        return 0

    def _info_bmi(self, x):
        if x < 18.5:
            return 1
        elif x >= 18.5 and x < 25.0:
            return 2
        elif x >= 25.0 and x < 30.0:
            return 3
        elif x >= 30.0 and x < 35.0:
            return 4
        elif x >= 35.0 and x < 40.0:
            return 5
        elif x >= 40.0:
            return 6
        return 0
    
    def _info_smoke(self, x):
        if x == 0:
            return 1
        elif x == 1: 
            return 2
        return 0
    
    def _info_drink(self, x):
        if x == 0:
            return 1
        elif x == 1: 
            return 2
        return 0


class MyDataSet2(Dataset):

    def __init__(self, dataframe, transform=None):
        self.transform = transform
        self.left_path_list = dataframe['left_eye'].to_list()
        self.right_path_list = dataframe['right_eye'].to_list()
        self.label_list = dataframe['label'].to_list()
        self.pd_data = dataframe

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        left_file = self.left_path_list[index]
        right_file = self.right_path_list[index]
        left_path = os.path.join(left_file, 'RGB.png')
        right_path = os.path.join(right_file, 'RGB.png')
        Limg = Image.open(left_path)
        Rimg = Image.open(right_path)
        label = self.label_list[index]
        label = torch.tensor(int(label))
        clicinfo = torch.LongTensor([
            self._info_sex(self.pd_data.iloc[index]["SEX"]),
            self._info_age(self.pd_data.iloc[index]["AGENEW2014"]),
            self._info_bmi(self.pd_data.iloc[index]["BMI"]),
            self._info_smoke(self.pd_data.iloc[index]["SMOKE_CUR"]),
            self._info_drink(self.pd_data.iloc[index]["DRINK_CUR"]),
        ])
        if self.transform is not None:
            Limg = self.transform(Limg)
            Rimg = self.transform(Rimg)
        return Limg, Rimg, label, clicinfo
    
    def _info_sex(self, x):
        if x == 1: # mam
            return 1
        elif x == 2: # woman
            return 2
        return 0
    
    def _info_age(self, x):
        if x < 40: # young
            return 1
        elif x >= 40 and x < 60: # middle
            return 2
        elif x >= 60: # old
            return 3
        return 0

    def _info_bmi(self, x):
        if x < 18.5:
            return 1
        elif x >= 18.5 and x < 25.0:
            return 2
        elif x >= 25.0 and x < 30.0:
            return 3
        elif x >= 30.0 and x < 35.0:
            return 4
        elif x >= 35.0 and x < 40.0:
            return 5
        elif x >= 40.0:
            return 6
        return 0
    
    def _info_smoke(self, x):
        if x == 0:
            return 1
        elif x == 1: 
            return 2
        return 0
    
    def _info_drink(self, x):
        if x == 0:
            return 1
        elif x == 1: 
            return 2
        return 0
    

    
    
    


    
    
    







    
    
    

    
