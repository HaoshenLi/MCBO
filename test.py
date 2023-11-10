import numpy as np
import torch.nn as nn
import torch
import os
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics

from datasets.my_dataset import *
from models.model import *


def Find_Optimal_Cutoff(TPR, FPR, threshold):
	
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def test(model):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)

    test_csv_path = 'data_split/test_info.csv'
    train_csv_path = 'data_split/train_info.csv'

    data_transform = {
        "test": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.310, 0.386, 0.549], [0.228, 0.279, 0.340])])}

    test_dataset = MyDataSet(test_csv_path, transform=data_transform['test'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=4,
                                               shuffle=True)
    train_dataset = MyDataSet(train_csv_path, transform=data_transform['test'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4,
                                               shuffle=True)
    test_num = len(pd.read_csv(test_csv_path))
    train_num = len(pd.read_csv(train_csv_path))

    model.eval()

    label_list = []
    prob_list = []
    pred_list = []
    for i, (left_tensor, right_tensor, label, clicinfo) in enumerate(train_dataloader): 

        label = label.to(device)
        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        clicinfo = clicinfo.to(device)
       
        out = model(left_tensor, right_tensor, clicinfo=clicinfo)
        out = F.softmax(out, dim=1)

        label_list.extend(label.cpu().numpy())
        prob_list.extend(out[:, 1].detach().cpu().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
    optimal_threshold, _ = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    print('Train optimal_threshold:{:.4f}'.format(optimal_threshold))
        

    label_list = []
    prob_list = []
    pred_list = []
    for i, (left_tensor, right_tensor, label, clicinfo) in enumerate(test_dataloader): 

        label = label.to(device)
        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        clicinfo = clicinfo.to(device)
        
        out = model(left_tensor, right_tensor, clicinfo=clicinfo)
        out = F.softmax(out, dim=1)
        pred = torch.where(out[:, 1] > optimal_threshold, 1, 0)

        label_list.extend(label.cpu().numpy())
        prob_list.extend(out[:, 1].detach().cpu().numpy())
        pred_list.extend(pred.detach().cpu().numpy())
    
    fpr, tpr, thresholds = metrics.roc_curve(label_list, prob_list)
    optimal_threshold, _ = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    print('Test optimal_threshold:{:.4f}'.format(optimal_threshold))
    correct_num = np.sum(np.array(pred_list) == np.array(label_list))
    confusion = metrics.confusion_matrix(label_list, pred_list)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    sensitivity = round(TP / (TP + FN), 4)
    specificity = round(TN / (FP + TN), 4)
    print('Test Accucacy:{:3f}'.format(correct_num / test_num))
    print('AUC:{:.4f}'.format(metrics.roc_auc_score(label_list, prob_list)))
    print('sensitivity:{}'.format(sensitivity))
    print('specificity:{}'.format(specificity))


if __name__ =='__main__':
    model = MCBO()
    model_path = 'results/MCBO/lr:1e-6_layer_1_heads_1/model_best.pth'
    model.load_state_dict(torch.load(model_path))
    test(model)
    
   
    