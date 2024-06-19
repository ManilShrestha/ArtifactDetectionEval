import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from lib.CustomDataset import TimeSeriesHDF5Dataset
from torch.utils.data import DataLoader
from lib.VAE import VAE	
from lib.Utilities import *
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import yaml
import os
import wandb

torch.manual_seed(10)

############### SCRIPT VARIABLES #######################
segment_length_sec = config['segment_length_sec']
sampling_rate = config['sampling_rate']
overlap = config['overlap']

latent_dim = 5
lr = 1e-3
epochs = 50
batch_size=64

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
best_model_path = 'models/deep_clean_abp_best.pt'
directory_path = '/storage/ms5267@drexel.edu/precicecap_downloads/'
mode = ['ABP','ART']

#########################################################

percentile_threshold_list = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,95,99]


def test(test_file, vae=None):
	"""Tests the DeepClean VAE on a test file

	Args:
		test_file (str): a single test file used to test the model on
	"""
	if vae is None:
		vae = torch.load(best_model_path)

	vae.eval()

	mseloss = torch.nn.MSELoss(reduction = 'none')
	test_mseloss_arr, truth_label = [], []

	datafile = os.path.join(directory_path, test_file)
	
	for m in mode:
		dataset  = TimeSeriesHDF5Dataset(datafile, m, segment_length_sec, overlap)

		if len(dataset)!=0:    
			dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,shuffle=True, pin_memory=True)
			
			mean = torch.load(f'{config["stored_mean_abp"]}_{segment_length_sec}sec')
			std = torch.load(f'{config["stored_std_abp"]}_{segment_length_sec}sec')

			with torch.no_grad():
				for _, data, label, ts in tqdm(dataloader):
					data = data.unsqueeze(1).float().to(device)

					data = (data-mean)/std
					
					recon_batch, z_mean, z_log_var = vae(data)
					mseloss_val = mseloss(recon_batch, data)
					# Compute MSE loss for each instance in the batch
					mseloss_per_instance = mseloss_val.mean(dim=2).cpu().numpy()
					
					# Append MSE loss for each instance to the list
					test_mseloss_arr.extend(mseloss_per_instance)
					truth_label.extend(label.cpu().numpy())

	mseloss_arr = get_threshold(vae, train_files)

	for percentile_threshold in percentile_threshold_list:
		threshold = np.percentile(mseloss_arr, percentile_threshold)

		log_info(f'Threholding at {threshold} for percentile: {percentile_threshold}')

		pred_label = [0 if i[0] < threshold else 1 for i in test_mseloss_arr]

		log_info(f"Accuracy:{accuracy_score(pred_label, truth_label)}")
		log_info(f"F1: {f1_score(pred_label, truth_label)}")
		print(confusion_matrix(truth_label, pred_label))


def get_threshold(vae, train_files):
	"""Calculates threshold for anomaly detection

	Args:
		vae (VAE): trained VAE model
		train_files (list): list of training files
		percentile_threshold (int, optional): threshold percentile. Defaults to 90.

	Returns:
		float: calculated threshold
	"""
	MSELoss = torch.nn.MSELoss()
	mseloss_arr = []

	mean = torch.load(f'{config["stored_mean_abp"]}_{segment_length_sec}sec')
	std = torch.load(f'{config["stored_std_abp"]}_{segment_length_sec}sec')

	with torch.no_grad():
		for filename in train_files:
			log_info(f"Processing {filename}")
			datafile = os.path.join(directory_path, filename)

			for m in mode:
				# tqdm_loader = tqdm(train_loader, desc='Training Batch', unit='batch')
				dataset  = TimeSeriesHDF5Dataset(datafile, m, segment_length_sec, overlap)    
				
				if len(dataset)!=0:
					dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4,shuffle=True, pin_memory=True)

					for _, data, label, ts in dataloader:
						filter = filter_abp_batch(data, label)
						data = data.unsqueeze(1).float().to(device)[filter]
						if len(data)>0:
							data = (data - mean)/std
							recon_batch, z_mean, z_log_var = vae(data)
							mseloss = MSELoss(recon_batch, data).item()
							
							# Appending loss to analyze
							mseloss_arr.append(mseloss)

	return mseloss_arr


def compute_mean_std(train_files):
	"""Computes mean and standard deviation of the entire dataset.

	Args:
		train_files (list): list of training files.
	Returns:
		tuple: mean and standard deviation of the entire dataset.
	"""
	# Initialize sum and sum of squares
	sum_data = 0
	sum_sq_data = 0
	n = 0
	log_info("Computing mean and standard deviation from the training hdf5 files.")
	with torch.no_grad():
		for filename in tqdm(train_files):
			log_info(f"Processing {filename}")
			datafile = os.path.join(directory_path, filename)

			# Load the dataset
			for m in mode:
				dataset = TimeSeriesHDF5Dataset(datafile, m, segment_length_sec, overlap)
				if len(dataset)!=0:
					dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=True)

					# Loop through batches in the DataLoader
					for _, data, label, ts in dataloader:
						filter = filter_abp_batch(data, label)
						data = data[filter].float().to(device)  # No need to unsqueeze for single-channel data
						if len(data)>0:
							sum_data += torch.sum(data)
							sum_sq_data += torch.sum(data ** 2)
							n += data.numel()

	# Compute mean and standard deviation
	mean = sum_data / n
	std_dev = torch.sqrt((sum_sq_data / n) - (mean ** 2))

	return mean.item(), std_dev.item()


if __name__ == '__main__':
	train_files = [
					# '4_Patient_2022-02-05_08:59.h5'
					'34_Patient_2023-04-04_22:31.h5'
					# , '35_Patient_2023-04-03_19:51.h5'
					# , '50_Patient_2023-06-12_21:10.h5'
					# , '53_Patient_2023-06-25_21:39.h5'
					# , '55_Patient_2023-06-13_00:47.h5'
				  	# , '59_Patient_2022-01-31_23:19.h5'
					, '73_Patient_2017_Dec_18__11_19_55_297272.h5'
					, '74_Patient_2023-08-05_06:00.h5'
					, '85_Patient_2023-05-12_17:53.h5'
					# , '90_Patient_2023-03-21_12:19.h5' 
					# , '101_Patient_2023_Nov_9__22_24_41_155873.h5'
					# , '139_Patient_2024_Mar_4__7_32_51_662674.h5'
					]
	
	# test_file = '85_Patient_2023-05-12_17:53.h5'
	test_file = '90_Patient_2023-03-21_12:19.h5'
	
	# train_files = ['73_Patient_2017_Dec_18__11_19_55_297272.h5', '59_Patient_2022-01-31_23:19.h5', '74_Patient_2023-08-05_06:00.h5', '34_Patient_2023-04-04_22:31.h5']

	# test_file = '110_Patient_2023_Sep_28__23_52_07_705708.h5'

	# Set reset=True if you want to recalculate the mean and std of population.
	print(test(test_file))