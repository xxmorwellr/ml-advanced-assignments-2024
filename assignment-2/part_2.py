import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.transforms.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from wikiart import WikiArtDataset 

# Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # output_size (64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # output_size (128, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  #output_size (256, H/8, W/8)
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # output_size (128, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # output_size (64, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # output_size (3, H, W)
            nn.Sigmoid()  # Constrain the output between [0,1]
        )

    def forward(self, x):
        compressed = self.encoder(x)  # compressed representation
        reconstructed = self.decoder(compressed)  # reconstructing the image
        return compressed, reconstructed

# Train autoencoder
def train_autoencoder(dataset, device, epochs=2, batch_size=32, learning_rate=1e-3):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    autoencoder = Autoencoder().to(device)
    criterion = nn.MSELoss()  # Loss function
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0
        for images, _, _ in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            _, reconstructed = autoencoder(images)
            loss = criterion(reconstructed, images)  # bias between the input image and the reconstructed image
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Average loss: {total_loss/len(dataloader):.4f}')
    
    print("...finished!")
    
    return autoencoder

# Extracting the compressed representation
def extract_compressed_representations(autoencoder, dataset, device):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    compressed_representations = []
    labels = []
    
    with torch.no_grad():
        for images, label, _ in dataloader:
            images = images.to(device)
            compressed, _ = autoencoder(images)  # Extract the compressed representation of the encoder output
            compressed_representations.append(compressed.view(compressed.size(0), -1).cpu().numpy())  # Flatten to 1D
            labels.append(label.cpu().numpy())
    
    return np.vstack(compressed_representations), np.hstack(labels)

# Cluster and Visualize
def cluster_and_visualize(compressed_representations, labels, output_file='cluster_plot.png'):
    # cluster
    kmeans = KMeans(n_clusters=10, random_state=42) # default: n_clusters = n_labels
    clusters = kmeans.fit_predict(compressed_representations)

    # dimensionality reduction
    pca = PCA(n_components=2) # 2-D
    reduced_data = pca.fit_transform(compressed_representations)
    
    # visualize
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title('Clustering of Image Representations')
    
    # Save the figure to a file
    plt.savefig(output_file)
    print(f'Plot saved to {output_file}')

# Main
if __name__ == '__main__':
    imgdir = 'train'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load data
    dataset = WikiArtDataset(imgdir, device=device)

    # train autoencoder
    print("Training autoencoder...")
    autoencoder = train_autoencoder(dataset, device)

    # extract compressed representations
    print("Extract compressed representations...")
    compressed_representations, labels = extract_compressed_representations(autoencoder, dataset, device)

    # cluster and visualize
    print("Clustering and ploting...")
    cluster_and_visualize(compressed_representations, labels)
