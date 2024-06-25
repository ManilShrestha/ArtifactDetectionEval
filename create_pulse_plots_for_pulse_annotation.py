# This script creates the pulse level plots and also creates the info and data csv files for the pulse based on the pulse_id
import sys
# sys.path.append('..')

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
import datetime

random.seed(10)

##################################################################
# Out file for the csv info and raw data
info_out_file = 'data/pulse_plots/pulse_info.csv'
data_out_file = 'data/pulse_plots/pulse_data.csv'

directory_path = '/storage/ms5267@drexel.edu/precicecap_downloads/'
# First collect the artifact pulses
artifact_pulse_info = []
artifact_pulse_data = []

mode = ['ART','ABP']
tmp_info, tmp_data=[],[]
plot_id = 0

MAX_ARTIFACT_PLOTS_PER_PATIENT = 200
MAX_NON_ARTIFACT_PLOTS_PER_PATIENT = 100
##################################################################

hdf5_files = [
				'4_Patient_2022-02-05_08:59.h5'
				# , '34_Patient_2023-04-04_22:31.h5'
				# , '35_Patient_2023-04-03_19:51.h5'
				, '50_Patient_2023-06-12_21:10.h5'
				# , '53_Patient_2023-06-25_21:39.h5'
				# , '55_Patient_2023-06-13_00:47.h5'
				# , '59_Patient_2022-01-31_23:19.h5'
				
				# , '73_Patient_2017_Dec_18__11_19_55_297272.h5'
				, '74_Patient_2023-08-05_06:00.h5'
				# , '85_Patient_2023-05-12_17:53.h5'
				
				, '90_Patient_2023-03-21_12:19.h5' 
				# , '101_Patient_2023_Nov_9__22_24_41_155873.h5'
				# , '139_Patient_2024_Mar_4__7_32_51_662674.h5'
				, '110_Patient_2023_Sep_28__23_52_07_705708.h5'
			]
##################################################################

def get_pulses(signal, sigma=2, distance = 58):
	filtered_signal = gaussian_filter1d(signal, sigma=sigma)
	troughs, _ = scipy.signal.find_peaks(-filtered_signal, distance = distance) # At least 0.5 secs apart. 125Hz so 62 points
	return troughs, _

def plot_pulse_signal(signal, plot_id, type):
	"""Stores the plot of pulse signals in the plot_dir folder

	Args:
		signal (_type_): Signal for the pulse
		plot_id (_type_): Unique identifier for the plot
		type (_type_): 0 for good non-artifact and 1 for the artifacts
	"""
	if type==1:
		plot_dir = '/home/ms5267@drexel.edu/moberg-precicecap/ArtifactDetectionEval/data/pulse_plots/plot_images/artifacts/'
	else:
		plot_dir = '/home/ms5267@drexel.edu/moberg-precicecap/ArtifactDetectionEval/data/pulse_plots/plot_images/non_artifacts/'

	filename = plot_dir + str(plot_id) + '.png'
	# Check if the input is a PyTorch tensor and convert to NumPy array if necessary
	if isinstance(signal, torch.Tensor):
		signal = signal.numpy()
	
	# Plot the signal
	plt.figure(figsize=(10, 4))
	plt.plot(signal, linestyle='-', color='b')
	plt.xticks([])
	plt.ylabel('Amplitude')
	plt.title('Pulse Signal')
	plt.grid(True)
	
	plt.savefig(filename, bbox_inches='tight')
	
	plt.close()

def get_time_from_ts(ts):
	utc_time = datetime.datetime.fromtimestamp(int(ts)/1e6, tz=pytz.utc)
	eastern = pytz.timezone('US/Eastern')
	est_time = utc_time.astimezone(eastern)
	  
	return est_time

#############################################################################
# Loop to get artifact pulses
for filename in tqdm(hdf5_files):
	log_info(f"Processing {filename}")
	datafile = os.path.join(directory_path, filename)
	artifact_count = 0
	
	# Load the dataset
	for m in mode:
		dataset  = TimeSeriesHDF5Dataset(datafile, m, segment_len=2, overlap=0, phase="test", smoothen=False) 

		if len(dataset)==0:
			print("No data to process, continuing...")
			continue

		dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
		
		print(f"{len(dataset)} files to process...")

		for start_i, data, lbl, ts in tqdm(dataloader):
			# Filtering out the dead signals
			filter = filter_abp_batch_scae(data)
			
			start_i = start_i[filter]
			data = data[filter]
			lbl = lbl[filter]
			ts = ts[filter]

			if len(data)>0:
				for b_n in range(len(start_i)):
					if artifact_count>=MAX_ARTIFACT_PLOTS_PER_PATIENT:
						# raise BreakNestedLoops
						break
					
					start_idx = start_i[b_n]
					label = lbl[b_n]
					timestamp  = ts[b_n]
					signal_data = data[b_n]

					if label==1:
						pulses, _ = get_pulses(signal_data)
						# print(pulses, _)

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

							est_start_time = get_time_from_ts(start_ts.item()).strftime("%Y-%m-%d %H:%M:%S")

							pulse_signal_raw = signal_data[pulses[p]:pulses[p+1]]
							plot_pulse_signal(pulse_signal_raw, plot_id=plot_id, type=label)
							
							
							tmp_info = [plot_id, filename, m, start_artifact.item(), end_artifact.item(), start_ts.item(), est_start_time, label.item()]
							tmp_data = [plot_id, filename, m, start_artifact.item(), end_artifact.item(), start_ts.item(), est_start_time, label.item()]
							tmp_data.extend(pulse_signal_raw.numpy())
							plot_id+=1
							
							artifact_pulse_info.append(tmp_info)
							artifact_pulse_data.append(tmp_data)
		
	print(f"For {filename} there are {artifact_count} artifact pulses")
								

########################################################################################
# Loop to get the non-artifact pulses
for filename in tqdm(hdf5_files):
	log_info(f"Processing {filename}")
	datafile = os.path.join(directory_path, filename)
	non_artifact_count = 0
	
	# Load the dataset
	for m in mode:
		dataset  = TimeSeriesHDF5Dataset(datafile, m, segment_len=20, overlap=0, phase="train", smoothen=False) 

		if len(dataset)==0:
			print("No data to process, continuing...")
			continue

		dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True, pin_memory=True)
		
		print(f"{len(dataset)} files to process...")

		for start_i, data, lbl, ts in tqdm(dataloader):
			# Filtering out the dead signals
			filter = filter_abp_batch_scae(data)
			
			start_i = start_i[filter]
			data = data[filter]
			lbl = lbl[filter]
			ts = ts[filter]

			if len(data)>0:
				for b_n in range(len(start_i)):
					if non_artifact_count>=MAX_NON_ARTIFACT_PLOTS_PER_PATIENT:
						# raise BreakNestedLoops
						break
					
					start_idx = start_i[b_n]
					label = lbl[b_n]
					timestamp  = ts[b_n]
					signal_data = data[b_n]

					if label==0:
						pulses, _ = get_pulses(signal_data,distance=50)
						# print(pulses, _)

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
							
							
							if pulses[p+1]>=len(timestamp):
								print("Timestamp shorter than the requested index")
								end_ts = timestamp[-1]
							else:
								end_ts = torch.tensor(-1)

							est_start_time = get_time_from_ts(start_ts.item()).strftime("%Y-%m-%d %H:%M:%S")

							pulse_signal_raw = signal_data[pulses[p]:pulses[p+1]]
							plot_pulse_signal(pulse_signal_raw, plot_id=plot_id, type=label)
							
							
							tmp_info = [plot_id, filename, m, start_artifact.item(), end_artifact.item(), start_ts.item(), est_start_time, label]
							tmp_data = [plot_id, filename, m, start_artifact.item(), end_artifact.item(), start_ts.item(), est_start_time, label]
							tmp_data.extend(pulse_signal_raw.numpy())
							plot_id+=1
							
							artifact_pulse_info.append(tmp_info)
							artifact_pulse_data.append(tmp_data)
		
	print(f"For {filename} there are {non_artifact_count} non-artifact pulses")


with open(info_out_file, 'w', newline='') as file:
	# Create a CSV writer
	writer = csv.writer(file)
	
	# Write the data to the CSV file
	writer.writerows(artifact_pulse_info)

	print(f"Data written in: {info_out_file}")

with open(data_out_file, 'w', newline='') as file:
	# Create a CSV writer
	writer = csv.writer(file)
	
	# Write the data to the CSV file
	writer.writerows(artifact_pulse_data)

	print(f"Data written in: {data_out_file}")