import sys
sys.path.append('..')

from lib.CustomDataset import TimeSeriesHDF5Dataset
from lib.Utilities import *
from torch.utils.data import DataLoader
import torch

import scipy
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import csv
import random

import signal
import os

###### SOMEONE IS KILLING MY PROCESS ###########

def sigterm_handler(signum, frame):
    print(f"Termination request from PID {os.getppid()}")
    sys.exit(1)

signal.signal(signal.SIGTERM, sigterm_handler)
################################################


directory_path = '/storage/ms5267@drexel.edu/precicecap_downloads/'

hdf5_files = [
				'4_Patient_2022-02-05_08:59.h5'
				, '34_Patient_2023-04-04_22:31.h5'
				, '35_Patient_2023-04-03_19:51.h5'
				, '50_Patient_2023-06-12_21:10.h5'
				, '53_Patient_2023-06-25_21:39.h5'
				, '55_Patient_2023-06-13_00:47.h5'
				, '59_Patient_2022-01-31_23:19.h5'
				
				, '73_Patient_2017_Dec_18__11_19_55_297272.h5'
				, '74_Patient_2023-08-05_06:00.h5'
				, '85_Patient_2023-05-12_17:53.h5'
				
				, '90_Patient_2023-03-21_12:19.h5' 
				, '101_Patient_2023_Nov_9__22_24_41_155873.h5'
				, '139_Patient_2024_Mar_4__7_32_51_662674.h5'
				, '110_Patient_2023_Sep_28__23_52_07_705708.h5'
			]

def get_pulses(signal, sigma=2):
	filtered_signal = gaussian_filter1d(signal, sigma=sigma)
	troughs, _ = scipy.signal.find_peaks(-filtered_signal, distance = 30) # At least 0.5 secs apart. 125Hz so 62 points
	return troughs

# Open the CSV file
out_file = 'data/SCAE_ABP_indices.csv'
# First collect the artifact pulses
artifact_pulse_indices = []

mode = ['ART','ABP']
for filename in tqdm(hdf5_files):
	log_info(f"Processing {filename}")
	datafile = os.path.join(directory_path, filename)

	# Load the dataset
	for m in mode:
		dataset  = TimeSeriesHDF5Dataset(datafile, m, segment_len=2, overlap=0, phase="train", smoothen=False) 

		if len(dataset)==0:
			print("No data to process, continuing...")
			continue

		dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False, pin_memory=True)

		artifact_count, non_artifact_count= 0,0
		
		print(f"{len(dataset)} files to process...")

		for start_i, data, lbl, ts in tqdm(dataloader):
			filter = filter_abp_batch_scae(data)
			
			start_i = start_i[filter]
			data = data[filter]
			lbl = lbl[filter]
			ts = ts[filter]

			if len(data)>0:
				for b_n in range(len(start_i)):
					start_idx = start_i[b_n]
					label = lbl[b_n]
					timestamp  = ts[b_n]
					signal_data = data[b_n]

					if label==1:
						pulses = get_pulses(signal_data)

						if len(pulses)==0:
								continue
						for p in range(len(pulses)-1):
							artifact_count+=1
							start_artifact = pulses[p]+start_idx
							end_artifact = pulses[p+1]+start_idx

							if pulses[p]>=len(timestamp):
								print("Timestamp shorter than the requested index")
								start_ts = torch.tensor(-1)
							else:
								start_ts = timestamp[pulses[p]]
							
							
							if pulses[p+1]>=len(timestamp):
								print("Timestamp shorter than the requested index")
								end_ts = timestamp[-1]
							else:
								end_ts = torch.tensor(-1)

							tmp = [filename, m, start_artifact.item(), end_artifact.item(), start_ts.item(), end_ts.item(),label.item()]
							
							artifact_pulse_indices.append(tmp)
		
		dataloader2 = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
		

		for start_i, data, lbl, ts in tqdm(dataloader2):
			if non_artifact_count>=artifact_count:
				break

			filter = filter_abp_batch_scae(data)
			
			start_i = start_i[filter]
			data = data[filter]
			lbl = lbl[filter]
			ts = ts[filter]

			if len(data)>0:
				for b_n in range(len(start_i)):
					if non_artifact_count>=artifact_count:
						break
					start_idx = start_i[b_n]
					label = lbl[b_n]
					timestamp  = ts[b_n]
					signal_data = data[b_n]

					if label==0:
						pulses = get_pulses(signal_data)
						if len(pulses)==0:
							continue
						for p in range(len(pulses)-1):
							non_artifact_count+=1
							
							start_artifact = pulses[p]+start_idx
							end_artifact = pulses[p+1]+start_idx

							if pulses[p]>=len(timestamp):
								print("Timestamp shorter than the requested index")
								start_ts = torch.tensor(-1)
							else:
								start_ts = timestamp[pulses[p]]

							# start_ts = timestamp[pulses[p]]
							
							if pulses[p+1]>=len(timestamp):
								print("Timestamp shorter than the requested index")
								end_ts = torch.tensor(-1)
							else:
								end_ts = timestamp[pulses[p+1]]

							tmp = [filename, m, start_artifact.item(), end_artifact.item(), start_ts.item(), end_ts.item(),label.item()]
							
							artifact_pulse_indices.append(tmp)

		print(f"For {filename} {m}. Total artifact pulses: {artifact_count} and total non-artifact pulses: {non_artifact_count}")
		


print(f"Writing into file {out_file}. There are {len(artifact_pulse_indices)} rows.")
with open(out_file, 'w', newline='') as file:
	# Create a CSV writer
	writer = csv.writer(file)
	
	# Write the data to the CSV file
	writer.writerows(artifact_pulse_indices)

	print(f"Data written in: {out_file}")