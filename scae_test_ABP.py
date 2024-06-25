import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from lib.CustomDataset import PulseFromHDF5Dataset
from torch.utils.data import DataLoader
from lib.VAE import VAE	
from lib.Utilities import *
import torch.optim as optim
import torch.nn as nn

import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from lib.SCAE_models import StackedConvAutoencoder, ArtifactCNN

torch.manual_seed(10)
############### SCRIPT VARIABLES #######################


lr = 1e-4
epochs = 100
batch_size=32

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
best_ae_model_path = 'models/scae_abp_best_ae.pt'
best_cnn_model_path = 'models/scae_abp_best_cnn.pt'

mode = ['ART','ABP']



def CNN_val(val_file, model=None):
	if not model:
		model = torch.load(best_cnn_model_path)
	model.eval()
	model.to(device)

	SCAEModel = torch.load(best_ae_model_path)
	SCAEModel.to(device)
	SCAEModel.eval()
	
	truth_labels, pred_labels = [],[]
	criterion = nn.CrossEntropyLoss()

	dataset_size,train_loss = 0,0

	for m in mode:
		log_info(f"Performing validation on {val_file}. Mode: {m}")				
		# tqdm_loader = tqdm(train_loader, desc='Training Batch', unit='batch')
		dataset  = PulseFromHDF5Dataset(val_file, m, phase='test')

		if len(dataset)!=0:
			dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)
			with torch.no_grad():
				for _, data, label, _, _ in tqdm(dataloader):
					inputs = data.to(device).float()
					label = label.to(device)
					# Forward pass
					inputs_recon = SCAEModel(inputs)
					# Forward + backward + optimize
					outputs = model(inputs_recon)

					loss = criterion(outputs, label)
					train_loss += loss.item()
					dataset_size += len(data)

						# Convert outputs probabilities to predicted class
					_, predicted = torch.max(outputs.data, 1)

					truth_labels.extend(label.cpu().tolist())
					pred_labels.extend(predicted.tolist())
	
	acc, f1, cm = accuracy_score(truth_labels, pred_labels), f1_score(truth_labels, pred_labels), confusion_matrix(truth_labels, pred_labels)

	avg_train_loss = train_loss / dataset_size

	print(cm)
	model.train()

	return acc,f1,cm,avg_train_loss


if __name__ == '__main__':

	
	# validation_file = '110_Patient_2023_Sep_28__23_52_07_705708.h5'
	validation_file = '85_Patient_2023-05-12_17:53.h5'
	# validation_file = '4_Patient_2022-02-05_08:59.h5'

	print(CNN_val(validation_file, model=None))
