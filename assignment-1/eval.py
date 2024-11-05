import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import ThaiCharCNN
from dataloader import ThaiCharDataset
from train import load_label_mapping
from sklearn.metrics import precision_score, recall_score, f1_score

# Goal of this file: evaluate the model

def test_model(model, device, test_loader):
    print("Testing model...")

    model.eval()  
    test_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []

    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # collect all labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # calculate
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

def load_model(model_path, num_classes, device):
    model = ThaiCharCNN(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only='True'))
    model.to(device)
    return model

def main():
    label_mapping = load_label_mapping()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. load dataset split by dataloader.py
    test_dataset = torch.load('test_dataset.pth', weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 2. test model
    num_classes = len(label_mapping)
    model = load_model("thai_char_cnn.pth", num_classes, device)
    test_model(model, device, test_loader)

if __name__ == "__main__":
    main()
