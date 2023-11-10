import numpy as np
import torch.nn as nn
import torch
import os
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from sklearn import metrics

from datasets.my_dataset import *
from models.model import *
from utils.build_logging import build_logging
from utils.plot import plot_function
from config.config_train import *

       
def train(args):

    save_dir = args.save_dir
    model = args.model
    lr = args.lr
    batch_size = args.batch_size
    epoches = args.epoches
    milestones = args.milestones
    gamma = args.gamma
    which_eye = args.which_eye

    assert which_eye in ['left', 'right', 'union']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    os.makedirs(save_dir, exist_ok=True)
    logger = build_logging(os.path.join(save_dir, 'log.txt'), save_dir)
    printer = logger.info

    model.to(device)
    model_best_save_path = os.path.join(save_dir, 'model_best.pth')
    model_latest_save_path = os.path.join(save_dir, 'model_latest.pth')

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "linear" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))
    pg = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.CrossEntropyLoss().to(device) 
    optimizer = optim.AdamW(pg, lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    data_transform = {
        "train":transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.310, 0.386, 0.549], [0.228, 0.279, 0.340])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.310, 0.386, 0.549], [0.228, 0.279, 0.340])])}

    train_csv_path = args.train
    val_csv_path = args.val
    train_dataset = MyDataSet(train_csv_path, transform=data_transform['train'])
    val_dataset = MyDataSet(val_csv_path, transform=data_transform['val'])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    train_num = len(pd.read_csv(train_csv_path)['left_eye'].to_list())
    val_num = len(pd.read_csv(val_csv_path)['left_eye'].to_list())

    train_losses = []
    train_acc = []
    train_auc_list = []
    val_losses = []
    val_acc = []
    val_auc_list = []
    best_auc = 0

    for epoch in range(epoches): 
        printer('Epoch:{}'.format(epoch + 1))
        train_loss = 0
        correct_num = 0
        label_list = []
        prob_list = []
        pred_list = []

        model.train()
        for i, (left_tensor, right_tensor, label, clicinfo) in enumerate(train_dataloader): 
            
            batch = left_tensor.shape[0]
            label = label.to(device)
            left_tensor = left_tensor.to(device)
            right_tensor = right_tensor.to(device)
            clicinfo = clicinfo.to(device)
            
            if which_eye == 'left':
                out = model(left_tensor, clicinfo=clicinfo)
            elif which_eye == 'right':
                out = model(right_tensor, clicinfo=clicinfo)
            elif which_eye == 'union':
                out = model(left_tensor, right_tensor, clicinfo=clicinfo)
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 

            train_loss += (loss.detach().cpu().numpy() * batch)
            _, pred = out.max(1) 
            pred_list.extend(pred.detach().cpu().numpy())
            prods = F.softmax(out, dim=1)
            label_list.extend(label.cpu().numpy())
            prob_list.extend(prods[:, 1].detach().cpu().numpy())

        correct_num = np.sum(np.array(pred_list) == np.array(label_list))
        train_auc = metrics.roc_auc_score(label_list, prob_list)
        printer('epoch:{}\tTrain Loss:{:3f}\tTrain Auc:{:3f}\tTrain Accucacy:{:3f}'.format(epoch+1, train_loss / train_num, train_auc, correct_num / train_num))
        train_losses.append(train_loss / train_num)
        train_acc.append(correct_num / train_num)
        train_auc_list.append(train_auc)
        scheduler.step()

        model.eval()
        val_loss = 0
        val_correct_num = 0
        label_list = []
        prob_list = []
        pred_list = []

        for left_tensor, right_tensor, label, clicinfo in val_dataloader:
            
            batch = left_tensor.shape[0]
            left_tensor = left_tensor.to(device)
            right_tensor = right_tensor.to(device)
            label = label.to(device)
            clicinfo = clicinfo.to(device)

            if which_eye == 'left':
                out = model(left_tensor, clicinfo=clicinfo)
            elif which_eye == 'right':
                out = model(right_tensor, clicinfo=clicinfo)
            elif which_eye == 'union':
                out = model(left_tensor, right_tensor, clicinfo=clicinfo)
            loss = criterion(out, label)

            val_loss += (loss.detach().cpu().numpy() * batch)
            _, pred = out.max(1) 
            pred_list.extend(pred.detach().cpu().numpy())
            prods = F.softmax(out, dim=1)
            label_list.extend(label.cpu().numpy())
            prob_list.extend(prods[:, 1].detach().cpu().numpy())

        val_correct_num = np.sum(np.array(pred_list) == np.array(label_list))
        val_auc = metrics.roc_auc_score(label_list, prob_list)
        if val_auc > best_auc:
            torch.save(model.state_dict(), model_best_save_path) 
            best_auc = val_auc
        printer('epoch:{}\tVal Loss:{:3f}\tVal Auc:{:3f}\tVal Accucacy:{:3f}\tBest Val Auc:{:3f}'.format(epoch+1, val_loss / val_num, val_auc, val_correct_num / val_num, best_auc))
        val_losses.append(val_loss / val_num)
        val_acc.append(val_correct_num / val_num)
        val_auc_list.append(val_auc)
    
    torch.save(model.state_dict(), model_latest_save_path)
    
    plot_function(train_losses, val_losses, epoches, type='loss', save_path=save_dir)
    plot_function(train_acc, val_acc, epoches, type='acc', save_path=save_dir)
    plot_function(train_auc_list, val_auc_list, epoches, type='auc', save_path=save_dir)


if __name__ == '__main__':

    train(config_MCBO())
    train(config_MCM())
    train(config_BFM())
    


    



    

    

    


    
    
    
    
    