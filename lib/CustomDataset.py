import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from lib.Utilities import *
from scipy.signal import resample

class TimeSeriesHDF5Dataset(Dataset):
    def __init__(self, file_path, dataset_name, ts_dataset_name, segment_len, overlap=0):
        """
        Args:
            file_path (str): Path to the HDF5 file.
            dataset_name (str): The name of the dataset in the HDF5 file.
            sampling_freq (int): The sampling frequency of the time series data.
            segment_len (int): The length of time (in seconds) of the segments to return.
        """
        # Open the file
        log_info(f'Reading {file_path}')
        self.hdf5_file = h5py.File(file_path, 'r')
        self.data = self.hdf5_file[dataset_name]
        timestamp = self.hdf5_file[ts_dataset_name]

        self.sampling_freq = round(get_sampling_freq(timestamp[0:10]))

        log_info(f'Sampling frequency for this file is: {self.sampling_freq}')
        if self.sampling_freq!=125:
            log_info(f'Frequency will be resampled to 125Hz.')

        self.file_path = file_path
        self.dataset_name = dataset_name
        self.segment_len = segment_len
        self.overlap = overlap
        self.segment_size = self.sampling_freq * segment_len 

        # Compute the total number of segments in the dataset
        self.total_segments = int((len(self.data) - self.segment_size)//(self.segment_size-(overlap * self.segment_size)))

        log_info(f'There are a total of : {self.total_segments} segments of {self.segment_len} seconds with overlap of {self.overlap*100}%')


    def __len__(self):
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
        
        if self.sampling_freq>130:

            number_of_samples = int(len(segment) * 125 / self.sampling_freq)
            segment = resample(segment, number_of_samples)

        # Convert to PyTorch tensor
        segment_tensor = torch.from_numpy(segment).float()

        return segment_tensor

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()

    def __del__(self):
        self.close()


