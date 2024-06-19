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
				# '110_Patient_2023_Sep_28__23_52_07_705708.h5'
				'85_Patient_2023-05-12_17:53.h5'
			]

def get_pulses(signal, sigma=2):
	filtered_signal = gaussian_filter1d(signal, sigma=sigma)
	troughs, _ = scipy.signal.find_peaks(-filtered_signal, distance = 30) # At least 0.5 secs apart. 125Hz so 62 points
	return troughs

# Open the CSV file
out_file = 'data/SCAE_ABP_indices_test.csv'
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
						artifact_count+=1
					if label==0:
						non_artifact_count+=1

					pulses = get_pulses(signal_data)

					if len(pulses)==0:
						continue

					for p in range(len(pulses)-1):
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
	
		print(f"For {filename} {m}. Total artifact pulses: {artifact_count} and total non-artifact pulses: {non_artifact_count}")
					


with open(out_file, 'w', newline='') as file:
	# Create a CSV writer
	writer = csv.writer(file)
	
	# Write the data to the CSV file
	writer.writerows(artifact_pulse_indices)

	print(f"Data written in: {out_file}")