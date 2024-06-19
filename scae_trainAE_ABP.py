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


latent_dim = 20
lr = 1e-3
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

def SCAE_train(train_files, test_file):
	"""Trains the DeepClean VAE

	Args:
		train_files (list): List of hdf5 files used for training purpose
		test_file (str): a single test file used to test the model on
	"""
	best_loss = float('inf')
	model = StackedConvAutoencoder()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr)
	model.to(device)
	
	log_info(f'Start training: {time.ctime()}')
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
						if len(data)>0:

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
		
		val_loss = SCAE_val(test_file, model)
		wandb.log({"val_loss": val_loss})
		
		# log metrics to wandb
		wandb.log({"loss": avg_train_loss})
		
		log_info(f"Epoch {epoch}, Train Loss: {avg_train_loss}, Validation Loss: {val_loss}")

	time_end_train = time.time()
	log_info(f'Training finished:{time.ctime()}')

	log_info(f'Training time:{time_end_train - time_start_train}')
	wandb.finish()


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



if __name__ == '__main__':
	train_files = [
					'4_Patient_2022-02-05_08:59.h5'
					, '34_Patient_2023-04-04_22:31.h5'
					, '35_Patient_2023-04-03_19:51.h5'
					, '50_Patient_2023-06-12_21:10.h5'
					, '53_Patient_2023-06-25_21:39.h5'
					# , '55_Patient_2023-06-13_00:47.h5'
				  	# , '59_Patient_2022-01-31_23:19.h5'

					# , '73_Patient_2017_Dec_18__11_19_55_297272.h5'
					# , '74_Patient_2023-08-05_06:00.h5'
					# , '85_Patient_2023-05-12_17:53.h5'
					
					
					# , '90_Patient_2023-03-21_12:19.h5' 
					# , '101_Patient_2023_Nov_9__22_24_41_155873.h5'
					# , '139_Patient_2024_Mar_4__7_32_51_662674.h5'
					]
	
	# test_file = '85_Patient_2023-05-12_17:53.h5'
	test_file = '90_Patient_2023-03-21_12:19.h5'
	
	# train_files = ['73_Patient_2017_Dec_18__11_19_55_297272.h5', '59_Patient_2022-01-31_23:19.h5', '74_Patient_2023-08-05_06:00.h5', '34_Patient_2023-04-04_22:31.h5']

	# test_file = '110_Patient_2023_Sep_28__23_52_07_705708.h5'

	# Set reset=True if you want to recalculate the mean and std of population.
	SCAE_train(train_files, test_file)