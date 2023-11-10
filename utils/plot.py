import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_function(train_list, val_list, epoches, type, save_path):
	epoches_list = [i + 1 for i in range(epoches)]
	plt.plot(epoches_list, train_list, 'r')
	plt.plot(epoches_list, val_list, 'b')
	plt.legend(['train', 'val'])
	plt.xlabel('epoch')
	assert type == 'loss' or type == 'acc' or type == 'auc', 'type false'
	if type == 'loss':
		plt.ylabel('loss')
		plt.title('loss')
		plt.savefig(os.path.join(save_path, 'loss.png'))
	elif type == 'acc':
		plt.ylabel('acc')
		plt.title('acc')
		plt.savefig(os.path.join(save_path, 'acc.png'))
	elif type == 'auc':
		plt.ylabel('auc')
		plt.title('auc')
		plt.savefig(os.path.join(save_path, 'auc.png'))
	plt.close()
