import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from train import ThaiCharCNN
from dataloader import ThaiCharDataset
from train import load_label_mapping


## For new testing datasets with brand different structures, try to combine OpenCV to preprocess images
# not a complete script, try to construct an analysis logic and list obstacles
# BW vs. Gray: BW datasets has been binarized, while Gray datasets not

def load_model(model_path, num_classes, device):
    model = ThaiCharCNN(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only='True'))
    model.to(device)
    return model

## define preprocess logic
def preprocess_image(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image at {image_path} not found.")
    
    # Binarize the image (simple thresholding)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Detect contours (assume contours are words/characters)
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours top-to-bottom (for multi-line detection)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    sorted_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    return img, sorted_boxes

## extract texts 
def extract_text_from_boxes(img, boxes, model):
    results = []

    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    for box in boxes:
        x, y, w, h = box
        roi = img[y:y+h, x:x+w]  # Extract Region of Interest (ROI)
        roi_resized = cv2.resize(roi, (28, 28))  # Resize to fit model input size (example: 28x28)
        roi_tensor = transform(roi_resized).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            output = model(roi_tensor)
            predicted_label = torch.argmax(output, dim=1).item()  # Get predicted label
            
        results.append(predicted_label)
        
        # Optionally draw rectangle and label on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(predicted_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img, results

## collect predicted labels
def run_ocr(image_path, model):
    img, boxes = preprocess_image(image_path)
    annotated_img, predictions = extract_text_from_boxes(img, boxes, model)

    # Save the result with bounding boxes
    cv2.imwrite('annotated_image.png', annotated_img)
    
    print("Predicted labels:", predictions)

## load true labels corresponding to images
def load_ground_truths(directory):
    ground_truths = {}

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                label = f.read().strip()
                # match corresponding images
                image_name = os.path.splitext(filename)[0]
                ground_truths[image_name] = label

    return ground_truths

def main():

    # 1. Load the trained model
    label_mapping = load_label_mapping()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = len(label_mapping)
    model = load_model("thai_char_cnn.pth", num_classes, device)
    
    # 2. Extract predicted labels
    ## take 200dpi_BW as an example
    testingdir = "/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TestSet/Book/Image/200dpi_BW"
    for filename in os.listdir(testingdir):
        if filename.endswith('.bmp'): 
            image_path = os.path.join(testingdir, filename)
            print(f"processing: {image_path}")
            run_ocr(image_path, model)

    # 3. Compare with real char..how to map character to label_code?
    textdir = "/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TestSet/Book/Text"
    ground_truths = load_ground_truths(textdir)

if __name__ == "__main__":
    main()
