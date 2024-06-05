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
from sklearn.metrics import accuracy_score, f1_score
import yaml
import os
import wandb

############### SCRIPT VARIABLES #######################
config_path = 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

segment_length_sec = config['segment_length_sec']
sampling_rate = config['sampling_rate']
overlap = config['overlap']
datafile = '/storage/ms5267@drexel.edu/precicecap_downloads/90_Patient_2023-03-21_12:19.h5'
latent_dim = 25
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
lr = 1e-4
epochs = 100
best_model_path = 'models/deep_clean_abp_best.pt'
#########################################################

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="deepclean-all",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "VAE",
    "dataset": "PRECISECAP",
    "epochs": epochs,
    }
)


# Loss function
def vae_loss(recon_x, x, mu, log_var):    
	# print([len(i[0]) for i in x], [len(j[0]) for j in recon_x])
	BCE = F.mse_loss(recon_x, x, reduction='sum')
	KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
	return BCE + KLD


def train():
    # VAE init
    best_loss = float('inf')
    num_features = segment_length_sec * sampling_rate
    vae = VAE((1,num_features),latent_dim)
    vae.to(device)
    optimizer = optim.RMSprop(vae.parameters(), lr=lr, alpha=0.9, eps=1e-08, weight_decay=0.0)
    
    print('Start training:', time.ctime())
    time_start_train = time.time()

    for epoch in tqdm(range(epochs)):
        vae.train()
        train_loss = 0
        dataset_size = 0
        # Directory containing the HDF5 files
        directory_path = '/storage/ms5267@drexel.edu/precicecap_downloads/'

        for filename in os.listdir(directory_path):
            if filename.endswith('.h5'):
                log_info(f"Processing {filename}")
                datafile = os.path.join(directory_path, filename)
                
                # tqdm_loader = tqdm(train_loader, desc='Training Batch', unit='batch')
                dataset  = TimeSeriesHDF5Dataset(datafile, 'ABP', segment_length_sec, overlap)    
                dataloader = DataLoader(dataset, batch_size=32, num_workers=4,shuffle=True, pin_memory=True)
        
                for _, data in dataloader:
                    data = data.unsqueeze(1).float().to(device)
                    optimizer.zero_grad()
                    recon_batch, z_mean, z_log_var = vae(data)
                    loss = vae_loss(recon_batch, data, z_mean, z_log_var)
                    loss.backward()
                    train_loss += loss.item()
                    dataset_size += len(dataloader.dataset)
                    optimizer.step()
                                
        # Calculate average loss over the dataset
        avg_train_loss = train_loss / dataset_size

        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(vae, best_model_path)
        
        # log metrics to wandb
        wandb.log({"loss": avg_train_loss})
        
        log_info(f"Epoch {epoch}, Train Loss: {avg_train_loss}")

    time_end_train = time.time()
    log_info(f'Training finished:{time.ctime()}')

    log_info(f'Training time:{time_end_train - time_start_train}')
    wandb.finish()

if __name__ == '__main__':
    train()