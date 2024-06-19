import sys
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

from lib.FE_ExtractFeatures import ExtractFeatures


segment_length_sec = 30
sampling_rate = config['sampling_rate']
overlap = 0.95
directory_path = config['hdf5_file_dir']
mode = ['ABP','ART']

hdf5_files = ['4_Patient_2022-02-05_08:59.h5']

out_file = 'data/FE_features_train.csv'


features_all = []

for filename in tqdm(hdf5_files):
	log_info(f"Processing {filename}")
	datafile = os.path.join(directory_path, filename)
	
	# Load the dataset
	for m in mode:
		dataset  = TimeSeriesHDF5Dataset(datafile, m, segment_len=segment_length_sec, overlap=overlap, phase="train", smoothen=False) 

		if len(dataset)==0:
			print("No data to process, continuing...")
			continue

		dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False, pin_memory=True)

		artifact_count, non_artifact_count= 0,0
		

		total_count =0
		for start_i, data, lbl, ts in tqdm(dataloader):
			filter = filter_abp_batch_scae(data)
			
			start_i = start_i[filter]
			data = data[filter]
			lbl = lbl[filter]
			ts = ts[filter]

			if len(start_i)>0:
				for b_n in range(len(start_i)):
					start_idx = start_i[b_n]
					label = lbl[b_n]
					timestamp  = ts[b_n]
					signal_data = data[b_n]

					if label==1:
						artifact_count+=1
						input_data = signal_data.unsqueeze(dim=0).numpy()
						features = ExtractFeatures(input_data).get_features().squeeze()
						
						per_segment_features = [datafile, m, label.item()] + features.tolist()
						features_all.append(per_segment_features)
		
		# For non-artifact segments
		dataloader2 = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)

		for start_i, data, lbl, ts in tqdm(dataloader2):
			if non_artifact_count>=artifact_count:
				break
			filter = filter_abp_batch_scae(data)
			
			start_i = start_i[filter]
			data = data[filter]
			lbl = lbl[filter]
			ts = ts[filter]

			if len(start_i)>0:
				for b_n in range(len(start_i)):
					if non_artifact_count>=artifact_count:
						break
					start_idx = start_i[b_n]
					label = lbl[b_n]
					timestamp  = ts[b_n]
					signal_data = data[b_n]

					if label==0:
						non_artifact_count+=1
						input_data = signal_data.unsqueeze(dim=0).numpy()
						features = ExtractFeatures(input_data).get_features().squeeze()
						
						per_segment_features = [datafile, m, label.item()] + features.tolist()
						features_all.append(per_segment_features)
		
		print(f"For {filename} {m}. Total artifact pulses: {artifact_count} and total non-artifact pulses: {non_artifact_count}")


print(f"Writing into file {out_file}.")
with open(out_file, 'w', newline='') as file:
	# Create a CSV writer
	writer = csv.writer(file)
	
	# Write the data to the CSV file
	writer.writerows(artifact_pulse_indices)

	print(f"Data written in: {out_file}")