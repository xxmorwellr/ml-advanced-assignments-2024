import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from wikiart import WikiArtDataset
import matplotlib.pyplot as plt
import random

class ConditionalAutoencoder(nn.Module):
    def __init__(self, num_styles=27, style_embedding_dim=32): 
        super(ConditionalAutoencoder, self).__init__()
        
        # Style embedding
        self.style_embedding = nn.Embedding(num_styles, style_embedding_dim)

        # Encoder (same as before)
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + style_embedding_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, image, style):
        style_embed = self.style_embedding(style).view(style.size(0), -1, 1, 1)  # Reshape to fit convolution layers

        # Incorprate style in Encoder stage
        style_applied = style_embed.expand(image.size(0), -1, image.size(2), image.size(3))
        conditioned_input = torch.cat([image, style_applied], dim=1)

        # Encoder
        compressed = self.encoder(conditioned_input)

        # # Concatenate style embedding with compressed image representation
        # combined = torch.cat([compressed, style_embed.expand_as(compressed)], dim=1)

        # Decoder
        reconstructed = self.decoder(compressed)
        return reconstructed

# Train the conditional autoencoder
def train_conditional_autoencoder(dataloader, device, epochs=20, learning_rate=1e-3):
    autoencoder = ConditionalAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    print("Start training...")
    for epoch in range(epochs):
        total_loss = 0
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            reconstructed = autoencoder(images, labels)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], average loss: {total_loss/len(dataloader):.4f}')
    
    print("...finished!")

    return autoencoder


def test_autoencoder(autoencoder, test_images, test_styles, device, output_file):

    autoencoder.eval()
    
    test_images = test_images / 255.0 ## for read_image, range [0, 255]
    test_images = test_images.to(device)
    test_styles = test_styles.to(device)

    # reconstruct images with mismatched styles
    print("Generating reconstructed image with mismatched styles...")
    with torch.no_grad():
        reconstructed_images = autoencoder(test_images, test_styles)
    # restrict reconstructed_images within [0, 1]
    reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
    
    # move to CPU for visualization
    test_images = test_images.cpu()
    reconstructed_images = reconstructed_images.cpu()
    
    # plot
    print("Ploting contrast image...")
    fig, axes = plt.subplots(2, len(test_images), figsize=(12, 4))
    for i in range(len(test_images)):
        # original images
        axes[0, i].imshow(test_images[i].permute(1, 2, 0))  # change channel from (C, H, W)(Pytorch) to (H, W, C)(imshow)
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')

        # reconstructed images
        axes[1, i].imshow(reconstructed_images[i].permute(1, 2, 0))
        axes[1, i].axis('off')
        axes[1, i].set_title('Reconstructed')

    # Save the figure to a file
    plt.savefig(output_file)
    print(f'Plot saved to {output_file}')

# Main
if __name__ == '__main__':
    imgdir = 'train'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = WikiArtDataset(imgdir, device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train encoder
    autoencoder = train_conditional_autoencoder(dataloader, device=device, epochs=3, learning_rate=0.0001)

    # Collect some test images from dataloader
    test_images, test_styles, _ = next(iter(dataloader))

    # Make sure unique styles 
    unique_styles = test_styles.unique()
    selected_images = []
    selected_styles = []

    for style in unique_styles[:5]: 
        idx = (test_styles == style).nonzero(as_tuple=True)[0][0]
        # selected_images.append(test_images[idx])
        selected_images.append(test_images[0])  # append the same test image
        selected_styles.append(test_styles[idx])
    
    # Shuffle selected_styles 
    random.shuffle(selected_styles)

    test_images = torch.stack(selected_images).to(device)
    test_styles = torch.stack(selected_styles).to(device)
 
    test_autoencoder(autoencoder, test_images[:5], test_styles[:5], device, "contrast_image.png") 