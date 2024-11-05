import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from dataloader import ThaiCharDataset
import json

## Goal of this file: define and train the model

# define the CNN model
class ThaiCharCNN(nn.Module):
    def __init__(self, num_classes):
        super(ThaiCharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 1 channel (grayscale), 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 32 filters
        self.fc1 = nn.Linear(32 * 16 * 16, 128)  # Flatten and fully connected layer with 128 units
        self.fc2 = nn.Linear(128, 64)  # Hidden layer with 64 units
        self.fc3 = nn.Linear(64, num_classes)  # Output layer
        
        # Define a nonlinear activation function (ReLU)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling layer
        self.dropout = nn.Dropout(0.25)  # Optional dropout for regularization

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten the tensor before the fully connected layers
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Optional dropout
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)  # Output
        return x


# Training function
def train_model(model, device, train_loader, val_loader, epochs=3, learning_rate=0.001):
    print("Training model...")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Use Adam optimizer
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            # print("images:",images)
            # print("labels:",labels)
            
            optimizer.zero_grad()  # Zero out gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {running_loss / len(train_loader):.4f}")
        
        # Validation loop (optional, but useful to track overfitting)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # Turn off gradients for validation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # print("Outputs shape:", outputs.shape)
                # print("Labels:", labels)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    print("Training finished!")


def load_label_mapping(filepath='label_mapping.json'):
    with open(filepath, 'r') as f:
        return json.load(f)
    

def main():
    
    label_mapping = load_label_mapping()
    # print("label_mapping:", label_mapping)

    # 0. Parse command line arguments
    parser = argparse.ArgumentParser(description="Select hyperparameters for training...")
    parser.add_argument('--epochs', type=int, default=3, help="Specify the number of epochs for training (default 3)")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Specify the learning rate for training (default 0.001)")
    args = parser.parse_args()
    
    epochs = args.epochs
    learning_rate = args.learning_rate

    # 1. load dataset split by dataloader.py
    train_dataset = torch.load('train_dataset.pth', weights_only=False)
    # print("Number of samples in train_dataset:", len(train_dataset))
    val_dataset = torch.load('val_dataset.pth', weights_only=False)

    # Specify the number of output classes
    num_classes = len(label_mapping)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2. Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ThaiCharCNN(num_classes).to(device)

    # 3. Train the model
    train_model(model, device, train_loader, val_loader, epochs=epochs, learning_rate=learning_rate)
    
    # Save the trained model to a file
    torch.save(model.state_dict(), "thai_char_cnn.pth")
    print("Model saved to thai_char_cnn.pth!")

if __name__ == "__main__":
    main()
