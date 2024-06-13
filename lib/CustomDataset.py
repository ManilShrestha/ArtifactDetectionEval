import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.Utilities import *
from scipy.signal import resample
import yaml

class TimeSeriesHDF5Dataset(Dataset):
    def __init__(self, file_path, mode, segment_len, overlap=0, smoothen=True):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            mode (str): ABP, ART or ECG. The mode of signal to extract
            segment_len (int): Length of signal in seconds.
            overlap (float): How much overlap is required. 0.5 means 50% of overlap from previous segment.
            smoothen (Bool): Should the returned signal be smoothen with Moving average filter.
        """
        # Open the file
        self.hdf5_file = h5py.File(file_path, 'r')
        self.mode_exists =True
        self.mode = mode
        if mode == 'ECG':
            dataset_name, ts_dataset_name = 'Waveforms/ECG_II','Waveforms/ECG_II_Timestamps'
        if mode == 'ABP':
            dataset_name, ts_dataset_name = 'Waveforms/ABP_na','Waveforms/ABP_na_Timestamps'
        if mode == 'ART':
            dataset_name, ts_dataset_name = 'Waveforms/ART_na','Waveforms/ART_na_Timestamps'

        # if mode == 'ABP':
        #     if 'Waveforms/ABP_na' in self.hdf5_file:
        #         dataset_name, ts_dataset_name = 'Waveforms/ABP_na','Waveforms/ABP_na_Timestamps'
        #     else:
        #         dataset_name, ts_dataset_name = 'Waveforms/ART_na','Waveforms/ART_na_Timestamps'
        if dataset_name not in self.hdf5_file:
            log_info(f"No {dataset_name} in the hdf5 file: {self.hdf5_file}.")
            self.mode_exists = False
        else:
            self.data = self.hdf5_file[dataset_name]
            self.timestamp = self.hdf5_file[ts_dataset_name]
            self.sampling_freq = round(get_sampling_freq(self.timestamp[0:10]))

            self.file_path = file_path
            self.dataset_name = dataset_name
            self.segment_len = segment_len
            self.overlap = overlap
            self.segment_size = self.sampling_freq * segment_len
            self.smoothen = smoothen

            # Compute the total number of segments in the dataset
            self.total_segments = int((len(self.data) - self.segment_size)//(self.segment_size-(overlap * self.segment_size)))

            self.segment_length_sec = config['segment_length_sec']

        # log_info(f'There are a total of : {self.total_segments} segments of {self.segment_len} seconds with overlap of {self.overlap*100}%')


    def __len__(self):
        if not self.mode_exists:
            return 0
        return self.total_segments

    def __getitem__(self, idx):
        # Calculate start and end indices of the segment
        start_idx = (idx * self.segment_size - int(self.overlap*self.segment_size)) if idx>0 else idx * self.segment_size
        end_idx = start_idx + self.segment_size

        # Ensure accessing the data does not exceed the actual data length
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = end_idx - self.segment_size

        # Load data segment
        segment = self.data[start_idx:end_idx]
        timestamp_segment = self.timestamp[start_idx:end_idx]

        is_outlier = check_outlier(segment, self.mode)

        if not is_outlier:
            label = 1 if is_artifact_overlap(self.file_path, self.mode, [start_idx,end_idx]) else 0
        else:
            label = 1

        if self.sampling_freq!=config['sampling_rate']:
            number_of_samples = int(len(segment) * config['sampling_rate'] / self.sampling_freq)
            segment = resample(segment, number_of_samples)

        if self.smoothen:
            segment = moving_average_filter(segment, window_size=3)


        # Convert to PyTorch tensor
        segment_tensor = torch.from_numpy(segment).float()

        return start_idx, segment_tensor, label, timestamp_segment

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()

    def __del__(self):
        self.close()



