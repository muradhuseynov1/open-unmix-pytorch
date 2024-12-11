import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim=2974, latent_dim=32):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # dims = [2379, 1788, 1192, 596]
        dims = [24000, 16000, 8000, 2048]
        

        # Encoder: Five fully connected layers with decreasing dimensions
        self.fc1 = nn.Linear(input_dim, dims[0])
        self.fc2 = nn.Linear(dims[0], dims[1])
        self.fc3 = nn.Linear(dims[1], dims[2])
        self.fc4 = nn.Linear(dims[2], dims[3])
        self.fc5 = nn.Linear(dims[3], latent_dim * 2)  # For mean and logvar

        # Decoder: Five fully connected layers with increasing dimensions
        self.fc6 = nn.Linear(latent_dim, dims[3])
        self.fc7 = nn.Linear(dims[3], dims[2])
        self.fc8 = nn.Linear(dims[2], dims[1])
        self.fc9 = nn.Linear(dims[1], dims[0])
        self.fc10 = nn.Linear(dims[0], input_dim)  # Reconstruction

    def encode(self, x):
        # Forward pass through the encoder
        # print(f"Input shape to encoder: {x.shape}")
        h1 = F.relu(self.fc1(x))
        # print(f"Shape after fc1: {h1.shape}")
        h2 = F.relu(self.fc2(h1))
        # print(f"Shape after fc2: {h2.shape}")
        h3 = F.relu(self.fc3(h2))
        # print(f"Shape after fc3: {h3.shape}")
        h4 = F.relu(self.fc4(h3))
        # print(f"Shape after fc4: {h4.shape}")
        h5 = self.fc5(h4)
        # print(f"Shape after fc5: {h5.shape}")

        # Latent distribution: mu and logvar
        mu, logvar = h5[:, :self.latent_dim], h5[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)  # Avoid extreme values
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z):
        # Forward pass through the decoder
        h6 = F.relu(self.fc6(z))
        h7 = F.relu(self.fc7(h6))
        h8 = F.relu(self.fc8(h7))
        h9 = F.relu(self.fc9(h8))
        x_recon = torch.tanh(self.fc10(h9))
        # x_recon = torch.sigmoid(self.fc10(h9))  # Sigmoid to match input range [0, 1]
        return x_recon

    def forward(self, x):
        # print(f"Input to VAE: {x.shape}")
        mu, logvar = self.encode(x)
        # print(f"Latent variables shape: mu={mu.shape}, logvar={logvar.shape}")
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        # print(f"Reconstructed output shape: {x_recon.shape}")
        return x_recon, mu, logvar   
    
def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='mean')  # MSE for audio
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return BCE + KL / x.size(0)  # Average KL over batch


# Custom Dataset to load audio files
class AudioDataset(Dataset):
    def __init__(self, audio_folder, transform=None):
        self.audio_folder = audio_folder
        self.files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):  # Fix method name from _getitem_ to __getitem__
        file_path = os.path.join(self.audio_folder, self.files[idx])
        audio, sample_rate = torchaudio.load(file_path)

        # Log initial shape
        # print(f"Original audio shape: {audio.shape}, sample rate: {sample_rate}")

        # Convert stereo to mono
        if audio.shape[0] == 2:
            audio = audio.mean(dim=0)
            # print(f"Converted to mono. Shape: {audio.shape}")

        # Resample audio if needed
        target_sample_rate = 16000  # Define your target sample rate
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            audio = resampler(audio)
            # print(f"Resampled audio to {target_sample_rate} Hz. Shape: {audio.shape}")

        # Truncate or pad audio to match target length
        target_length = 16000 * 5
        if audio.shape[0] > target_length:
            audio = audio[:target_length]  # Truncate
            # print(f"Truncated audio. Shape: {audio.shape}")
        elif audio.shape[0] < target_length:
            padding = target_length - audio.shape[0]
            audio = F.pad(audio, (0, padding))  # Pad with zeros
            # print(f"Padded audio. Shape: {audio.shape}")

        # Normalize audio to [-1, 1]
        eps = 1e-8  # Small epsilon value for numerical stability
        audio = audio / (torch.max(torch.abs(audio)) + eps)  # Normalize audio to [-1, 1]

        # Flatten audio
        audio = audio.flatten()
        # print(f"Flattened audio. Shape: {audio.shape}")

        return audio
