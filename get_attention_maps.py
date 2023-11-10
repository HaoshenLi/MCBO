from models.model import *
from datasets.my_dataset import *

import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import seaborn as sns


def get_attention_maps(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model.to(device)

    test_csv_path = 'data_split/test_info.csv'

    data_transform = {
        "test": transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.310, 0.386, 0.549], [0.228, 0.279, 0.340])])}

    test_dataset = MyDataSet2(test_csv_path, transform=data_transform['test'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               shuffle=True)
    test_num = len(pd.read_csv(test_csv_path))
    model.eval()

    for i, (left_tensor, right_tensor, label, clicinfo, left_path, right_path) in tqdm(enumerate(test_dataloader)): 

        label = label.to(device)
        left_tensor = left_tensor.to(device)
        right_tensor = right_tensor.to(device)
        clicinfo = clicinfo.to(device)
        
        out, attn_maps = model(left_tensor, right_tensor)  
        attn_maps = attn_maps.detach().cpu().numpy()
        attn_maps_left = attn_maps[:, 1:50]
        attn_maps_right = attn_maps[:, 50:99]
        attn_maps_left = (attn_maps_left - np.min(attn_maps_left)) / (np.max(attn_maps_left) - np.min(attn_maps_left)) 
        attn_maps_right = (attn_maps_right - np.min(attn_maps_right)) / (np.max(attn_maps_right) - np.min(attn_maps_right)) 
        attn_maps_left = attn_maps_left.reshape((7, 7))
        attn_maps_right = attn_maps_right.reshape((7, 7))

        left_path = left_path[0]
        right_path = right_path[0]
        idx = left_path.split('/')[-2]
        label = left_path.split('/')[-3]
        idx_folder = os.path.join('attn_maps', label, idx)
        os.makedirs(idx_folder, exist_ok=True)
        left_attn_path = os.path.join(idx_folder, 'left_attn.png')
        right_attn_path = os.path.join(idx_folder, 'right_attn.png')
        left_path_save = os.path.join(idx_folder, 'left.png')
        right_path_save = os.path.join(idx_folder, 'right.png')
        leftraw_path_save = os.path.join(idx_folder, 'left_raw.png')
        rightraw_path_save = os.path.join(idx_folder, 'right_raw.png')

        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(attn_maps_left, ax=ax, cmap='YlOrRd', annot=False)
        plt.savefig(left_attn_path)

        f, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(attn_maps_right, ax=ax, cmap='YlOrRd', annot=False)
        plt.savefig(right_attn_path)
        
        image_left = cv2.imread(left_path)
        cv2.imwrite(leftraw_path_save, image_left)
        heatmap_left = cv2.resize(attn_maps_left, (image_left.shape[1], image_left.shape[0]))
        heatmap_left = cv2.applyColorMap(np.uint8(heatmap_left * 255), cv2.COLORMAP_JET)
        result_left = cv2.addWeighted(image_left, 0.7, heatmap_left, 0.3, 0)
        cv2.imwrite(left_path_save, result_left)

        image_right = cv2.imread(right_path)
        cv2.imwrite(rightraw_path_save, image_right)
        heatmap_right = cv2.resize(attn_maps_right, (image_right.shape[1], image_right.shape[0]))
        heatmap_right = cv2.applyColorMap(np.uint8(heatmap_right * 255), cv2.COLORMAP_JET)
        result_right = cv2.addWeighted(image_right, 0.7, heatmap_right, 0.3, 0)
        cv2.imwrite(right_path_save, result_right)

        print()


if __name__ == '__main__':
    model = MCBO()
    model_path = 'results/MCBO/lr:1e-6_layer_1_heads_1/model_best.pth'
    model.load_state_dict(torch.load(model_path))
    get_attention_maps(model)