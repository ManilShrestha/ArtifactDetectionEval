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

	def convert_bytes(b):
		# Convert byte string literals found in the CSV to string removing b' and '
		if isinstance(b, bytes):
			return b.decode('utf-8').strip("b'").strip("'")
		return b.strip("b'").strip("'")
	# Load the CSV file, applying conversion to all columns
	df = pd.read_csv(ann_file_path, converters={i: convert_bytes for i in range(8)}, header=None)
	# Rename columns if needed (assuming you know what each column represents)
	df.columns = ['ID1', 'ID2', 'Session', 'Data_Type', 'Start_Time', 'End_Time', 'Signal_Type', 'Lead_Type']
	return df

def is_artifact_overlap(file_path, mode, candidate_interval):
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
		filter = ['ABP', 'ART', 'ART1', 'ART2']
	else:
		filter = 'ECG'
	
	# Filter the DataFrame
	filtered_df = annotation_df[annotation_df.iloc[:, -2].isin(filter)]
	# Extract the first two columns and convert to NumPy array
	artifact_arr = filtered_df.iloc[:, :2].astype(int).to_numpy()

	return has_artifact(candidate_interval, artifact_arr)



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
    start_idx = bisect.bisect_left(timestamp_list, start_timestamp)
    # Find the position for the end_timestamp, where it would be placed after any equal values.
    end_idx = bisect.bisect_right(timestamp_list, end_timestamp)  # Subtract 1 to include the end_timestamp itself if it's in the lists
    return start_idx, end_idx

def filter_batch(batch, filter_pos_pct=0.8):
	# Create a boolean tensor that is True where data is positive
	is_positive = batch > 0
	# Compute the proportion of positive values in each row
	proportion_positive = is_positive.float().mean(dim=1)
	# Filter rows where more than 90% of the values are positive
	
	return proportion_positive > filter_pos_pct