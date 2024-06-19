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
import wandb
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
#########################################################
# start a new wandb run to track this script
wandb.init(
	# set the wandb project where this run will be logged
	project="scae-abp",
	config={
	"learning_rate": lr,
	"architecture": "scae",
	"dataset": "PRECISECAP",
	"epochs": epochs,
	}
)

#########################################################

def SCAE_train(train_files, val_file):
	"""Trains the DeepClean VAE

	Args:
		train_files (list): List of hdf5 files used for training purpose
		val_file (str): a single test file used to test the model on
	"""
	best_loss = float('inf')
	model = StackedConvAutoencoder()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	
	model.to(device)
	
	log_info(f'Start autoencoder training: {time.ctime()}')
	time_start_train = time.time()

	for epoch in tqdm(range(epochs)):
		model.train()
		train_loss = 0
		dataset_size = 0
		# Directory containing the HDF5 files
		for m in mode:
			for filename in train_files:
				
				log_info(f"Processing {filename}. Mode: {m}")				
				# tqdm_loader = tqdm(train_loader, desc='Training Batch', unit='batch')
				dataset  = PulseFromHDF5Dataset(filename, m)

				if len(dataset)!=0:
					dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
					for _, data, _, _, _ in dataloader:
						inputs = data.to(device).float()
						# Zero the parameter gradients
						optimizer.zero_grad()
						
						# Forward pass
						outputs = model(inputs)
						loss = criterion(outputs, inputs)
						
						# Backward pass and optimize
						loss.backward()
						optimizer.step()
						
						train_loss += loss.item()
						dataset_size += len(data)
								
		# Calculate average loss over the dataset
		avg_train_loss = train_loss / dataset_size

		if avg_train_loss < best_loss:
			best_loss = avg_train_loss
			torch.save(model, best_ae_model_path)
		
		val_loss = SCAE_val(val_file, model)
		wandb.log({"val_loss": val_loss})
		
		# log metrics to wandb
		wandb.log({"loss": avg_train_loss})
		
		log_info(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {val_loss}")

	time_end_train = time.time()
	log_info(f'Training finished:{time.ctime()}')

	log_info(f'Training time:{time_end_train - time_start_train}')


def SCAE_val(val_file, model):
	"""Tests the DeepClean VAE on a test file

	Args:
		test_file (str): a single test file used to test the model on
	"""
	model.eval()
	criterion = nn.MSELoss()
	
	# Compute log(MSE) - ensure mse is non-zero and positive
	val_recon_loss_arr=0.0
	dataset_size=0

	for m in mode:
		dataset  = PulseFromHDF5Dataset(val_file, m)

		if len(dataset)!=0:    
			dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,shuffle=True, pin_memory=True)

			with torch.no_grad():
				for _, data, _, _, _ in dataloader:
					data = data.float().to(device)

					inputs = data.to(device).float()
					
					outputs = model(inputs)
					loss = criterion(outputs, inputs)
					
					val_recon_loss_arr += loss.item()
					dataset_size += len(data)

	val_recon_loss_arr /= dataset_size
	print(f'Validation Loss: {val_recon_loss_arr:.4f}')

	return val_recon_loss_arr


def CNN_train(train_files,val_file):

	model = ArtifactCNN(num_classes=2)
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.0001)

	SCAEModel = torch.load(best_ae_model_path)
	SCAEModel.to(device)

	SCAEModel.eval()

	num_epochs = 100  # Number of training epochs
	best_loss = float('inf')  # Track the best validation loss

	log_info(f'Start CNN training: {time.ctime()}')
	time_start_train = time.time()

	for epoch in tqdm(range(num_epochs)):
		model.train()
		train_loss = 0
		dataset_size = 0
		for m in mode:
			for filename in train_files:
				
				log_info(f"Processing {filename}. Mode: {m}")				
				# tqdm_loader = tqdm(train_loader, desc='Training Batch', unit='batch')
				dataset  = PulseFromHDF5Dataset(filename, m)

				if len(dataset)!=0:
					dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
					for _, data, label, _, _ in dataloader:
						inputs = data.to(device).float()
						label =label.to(device)
						# Zero the parameter gradients
						optimizer.zero_grad()
						
						# Forward pass
						inputs_recon = SCAEModel(inputs)

						outputs = model(inputs_recon)
						loss = criterion(outputs, label)
						
						# Backward pass and optimize
						loss.backward()
						optimizer.step()
						
						train_loss += loss.item()
						dataset_size += len(data)
								
		# Calculate average loss over the dataset
		avg_train_loss = train_loss / dataset_size

		if avg_train_loss < best_loss:
			best_loss = avg_train_loss
			torch.save(model, best_cnn_model_path)
		
		val_metrics = CNN_val(val_file, model)

		CNN_val_acc = val_metrics[0]
		CNN_val_loss = val_metrics[3]

		wandb.log({"CNN_val_acc": CNN_val_acc
			 	  ,"CNN_train_loss": avg_train_loss
				  ,"CNN_val_loss": CNN_val_loss
				  })
		
		log_info(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Validation Accuracy: {CNN_val_acc}")

	time_end_train = time.time()
	log_info(f'Training finished:{time.ctime()}')

	log_info(f'Training time:{time_end_train - time_start_train}')
	wandb.finish()


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
		dataset  = PulseFromHDF5Dataset(val_file, m)

		if len(dataset)!=0:
			dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)
			with torch.no_grad():
				for _, data, label, _, _ in dataloader:
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
	train_files = [
					'4_Patient_2022-02-05_08:59.h5'
					, '34_Patient_2023-04-04_22:31.h5'
					, '35_Patient_2023-04-03_19:51.h5'
					, '50_Patient_2023-06-12_21:10.h5'
					, '53_Patient_2023-06-25_21:39.h5'
					, '90_Patient_2023-03-21_12:19.h5' 

					# , '55_Patient_2023-06-13_00:47.h5'
				  	# , '59_Patient_2022-01-31_23:19.h5'

					# , '73_Patient_2017_Dec_18__11_19_55_297272.h5'
					# , '74_Patient_2023-08-05_06:00.h5'
					# , '85_Patient_2023-05-12_17:53.h5'
					
					
					# , '101_Patient_2023_Nov_9__22_24_41_155873.h5'
					# , '139_Patient_2024_Mar_4__7_32_51_662674.h5'
					]
	
	# validation_file = '110_Patient_2023_Sep_28__23_52_07_705708.h5'
	validation_file = '85_Patient_2023-05-12_17:53.h5'
	# validation_file = '90_Patient_2023-03-21_12:19.h5'
	
	# train_files = ['73_Patient_2017_Dec_18__11_19_55_297272.h5', '59_Patient_2022-01-31_23:19.h5', '74_Patient_2023-08-05_06:00.h5', '34_Patient_2023-04-04_22:31.h5']

	# test_file = '110_Patient_2023_Sep_28__23_52_07_705708.h5'

	
	SCAE_train(train_files, validation_file)
	CNN_train(train_files, validation_file)

	wandb.finish()