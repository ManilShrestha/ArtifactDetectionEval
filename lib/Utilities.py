from datetime import datetime
import numpy as np
import bisect


def log_info(log_message):
	print( datetime.now().strftime("%H:%M:%S"),":\t ", log_message , "\n")



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
