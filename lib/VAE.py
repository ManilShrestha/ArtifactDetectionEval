import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class VAE(nn.Module):
	def __init__(self, input_shape, latent_dim):
		
		super(VAE, self).__init__()
		self.latent_dim = latent_dim

		linear_dim = int(16 * np.prod(input_shape[1:]) // 25)
		
		# Encoder
		self.encoder = nn.Sequential(
			nn.Conv1d(in_channels=input_shape[0], out_channels=8, kernel_size=15, padding=(15 - 1) // 2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=5, padding=(5-1)//2),
			nn.Conv1d(8, 16, 15, padding=(15-1)//2),
			nn.ReLU(),
			nn.MaxPool1d(5, padding=(5-1)//2),
			nn.Dropout(0.1),
			nn.Conv1d(16, 16, 15, padding=(15-1)//2),
			nn.ReLU(),
			nn.Flatten(),
			nn.Dropout(0.1),
			nn.Linear(linear_dim, 16),
			nn.ReLU()
		)

		self.z_mean = nn.Linear(16, latent_dim)
		self.z_log_var = nn.Linear(16, latent_dim)

		# Decoder
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 16),
			nn.ReLU(),
			nn.Linear(16, linear_dim),
			nn.ReLU(),
			nn.Unflatten(1, (16, linear_dim//16)),
			nn.ConvTranspose1d(16, 16, 15, stride=5, padding=6, output_padding=4),
			nn.ReLU(),
			nn.ConvTranspose1d(16, 8, 15, stride=5, padding=6, output_padding=4),
			nn.ReLU(),
			nn.Conv1d(8, input_shape[0], 15, padding=1)
		)
	
	def encode(self, x):
		h = self.encoder(x)
		return self.z_mean(h), self.z_log_var(h)
	
	def reparameterize(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return mu + eps*std
	
	def decode(self, z):
		return self.decoder(z)

	def forward(self, x):
		mu, log_var = self.encode(x)
		z = self.reparameterize(mu, log_var)
		return self.decode(z), mu, log_var

