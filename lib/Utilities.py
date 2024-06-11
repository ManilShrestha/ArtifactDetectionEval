from datetime import datetime
import numpy as np
import bisect
import yaml


config_path = '/home/ms5267@drexel.edu/moberg-precicecap/ArtifactDetectionEval/config.yaml'
with open(config_path, 'r') as file:
	config = yaml.safe_load(file)

annotation_dir = config['annotation_dir']

def log_info(log_message):
	print( datetime.now().strftime("%H:%M:%S"),":\t ", log_message , "\n")


def load_annotation_file(ann_file_path):
	import pandas as pd

	# def convert_bytes(b):
	# 	# Convert byte string literals found in the CSV to string removing b' and '
	# 	if isinstance(b, bytes):
	# 		return b.decode('utf-8').strip("b'").strip("'")
	# 	return b.strip("b'").strip("'")
	# Load the CSV file, applying conversion to all columns
	# df = pd.read_csv(ann_file_path, converters={i: convert_bytes for i in range(8)}, header=None)
	df = pd.read_csv(ann_file_path)

	# df.columns = ['ID1', 'ID2', 'Session', 'Data_Type', 'Start_Time', 'End_Time', 'Signal_Type', 'Lead_Type']
	return df

def is_artifact_overlap(file_path, mode, candidate_idx):
	"""Finds if the given indices contain artifact or not

	Args:
		file_path (str): Path of the datafile, this is to get the name of file
		mode (str): Either ABP or ECG
		start_idx (int): Start index
		end_idx (int): End index
	"""
	import os
	file_name = os.path.basename(file_path)
	annotation_file_name = annotation_dir + file_name + '-annotations.csv'

	if not os.path.exists(annotation_file_name):
		return False

	annotation_df = load_annotation_file(annotation_file_name)

	if mode == 'ABP':
		filter = ['ABP']
	elif mode =='ART':
		filter = ['ART']
	elif mode == 'ECG':
		filter = ['ECG']
	
	
	# Filter the DataFrame
	filtered_df = annotation_df[annotation_df.iloc[:, -1].isin(filter)]
	# Extract the first two columns and convert to NumPy array
	artifact_arr = filtered_df.iloc[:, :2].astype(float).astype(int).to_numpy()

	# print(artifact_arr,[candidate_idx[0],candidate_idx[1]])

	return has_artifact([candidate_idx[0],candidate_idx[1]], artifact_arr)



def has_artifact(candidate_interval, artifacts):
	for artifact in artifacts:
		# Calculate the maximum start time and minimum end time between candidate_interval and artifact
		start_max = max(candidate_interval[0], artifact[0])
		end_min = min(candidate_interval[1], artifact[1])
		
		# Check for overlap
		if start_max < end_min:
			# If there is an overlap, return True
			return True
	
	# If no overlap is found with any artifact, return False
	return False


def has_overlap(candidate_interval, test_instances):
	
	test_set = np.array(test_instances)[:,0:2]
	
	for sample in test_set:
		# Calculate the maximum start time and minimum end time between candidate_interval and artifact
		start_max = max(candidate_interval[0], sample[0])
		end_min = min(candidate_interval[1], sample[1])
		
		# Check for overlap
		if start_max < end_min:
			# If there is an overlap, return True
			return True
	
	# If no overlap is found with any artifact, return False
	return False

	
def get_sampling_freq(timestamp_list):
	# Calculate differences between consecutive timestamps
	time_differences = np.diff(timestamp_list/1e6)

	# Calculate the average interval
	average_interval = np.mean(time_differences)

	# Calculate the sampling frequency
	sampling_frequency = 1 / average_interval

	return sampling_frequency


def find_idx_from_ts(timestamp_list, start_timestamp, end_timestamp):
	# start_idx = bisect.bisect_left(timestamp_list, start_timestamp)
	# # Find the position for the end_timestamp, where it would be placed after any equal values.
	# end_idx = bisect.bisect_right(timestamp_list, end_timestamp)  # Subtract 1 to include the end_timestamp itself if it's in the lists
	start_idx = np.searchsorted(timestamp_list, start_timestamp)
	end_idx = np.searchsorted(timestamp_list, end_timestamp)
	return start_idx, end_idx


def filter_abp_batch(batch, label, filter_pos_pct=0.8):
	"""
	Filters rows in the batch based on the proportion of positive values and label is 0
	"""
	# Create a boolean tensor that is True where data is positive
	is_positive = (batch > 30) & (batch<350)
	label_filter = (label==0)

	# Compute the proportion of positive values in each row
	proportion_positive = is_positive.float().mean(dim=1)

	# Filter rows where more than 90% of the values are positive
	flag =  proportion_positive > filter_pos_pct

	flag = flag & label_filter

	return flag


def filter_exclude_outliers(batch, label, filter_pos_pct=0.5):
	"""
	Filters rows in the batch based on the proportion of positive values and label is 0
	"""
	# Create a boolean tensor that is True where data is positive
	dead_filter = (batch > 30) & (batch<350)
	# label_filter = (label==0)

	# Compute the proportion of positive values in each row
	proportion_dead = dead_filter.float().mean(dim=1)

	# Filter rows where more than 90% of the values are positive
	flag =  proportion_dead > filter_pos_pct

	# flag = flag & label_filter

	return flag


def filter_ecg_batch(batch, label, filter_pos_pct=0.8):
	"""
	Filters rows in the batch based on the proportion of values>4 and label is zero.
	"""
	# Create a boolean tensor that is True where data is positive
	filter_lt4 = batch < 4
	label_filter = label==0

	# Compute the proportion of positive values in each row
	proportion_positive = filter_lt4.float().mean(dim=1)
	# Filter rows where more than 80% of the values are positive
	flag = proportion_positive > filter_pos_pct
	flag = flag & label_filter
	
	return flag


def check_outlier(segment, mode, pct=0.5):
	thresholds = {
		'ABP': (30, 350),
		'ECG': (-4, 4),
	}
	if mode not in thresholds:
		raise ValueError("Unsupported mode provided")
	
	low, high = thresholds[mode]
	is_outlier = (segment < low) | (segment > high)
	proportion_outlier = is_outlier.astype(float).mean(axis=0)

	return proportion_outlier > pct


def moving_average_filter(signal, window_size=5):
	# Calculate the number of pads to add on each side
	pad_size = window_size // 2

	# Pad the signal by repeating the edge values
	padded_signal = np.pad(signal, pad_size, mode='edge')
	
	# Compute the moving average using a uniform filter
	averaged_signal = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
	
	return averaged_signal