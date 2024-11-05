import os
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image # torchvision.io.read_image doesn't support .bmp files
import torchvision.transforms as transforms
import argparse
import json

class ThaiCharDataset(Dataset):
    def __init__(self, imgdir, specified_language, specified_dpi, specified_font_style, label_mapping, transform=None):
        self.imgdir = imgdir
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.label_mapping = label_mapping

        # iterate the directory to get files
        for root, _, files in os.walk(imgdir):
            path_parts = root.split(os.sep)  # Split the path to get its parts
            # print("path_parts:", path_parts)
        
            if len(path_parts) >= 4:
                language = path_parts[-4]
                char_code = path_parts[-3] 
                dpi = path_parts[-2]        
                font_style = path_parts[-1] # Font style (normal, bold, etc.) is the last part

                # Check if the language, DPI, and font style match the specified criteria
                language_matches = (specified_language == 'both' or language == specified_language)
                dpi_matches = (specified_dpi == 'all' or dpi == specified_dpi)
                font_style_matches = (specified_font_style == 'all' or font_style == specified_font_style)
                
                # If all conditions match (or are 'all'), process the files
                if language_matches and dpi_matches and font_style_matches:
                    for file in files:
                        if file.endswith(".bmp"): 
                            img_path = os.path.join(root, file)
                            # print("img_path:", img_path)
                            # print("char_code:", char_code)
                            self.img_paths.append(img_path)
                            self.img_labels.append(int(char_code))  # save character code as ints

        # Convert labels using the provided global mapping
        self.encoded_labels = [self.label_mapping[label] for label in self.img_labels]

        # convert
        self.encoded_labels = [self.label_mapping[label] for label in self.img_labels]
        # self.encoded_labels = torch.tensor(self.encoded_labels)
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path) # open images
        label_code = self.encoded_labels[idx] # get codes

        # apply transform
        if self.transform:
            image = self.transform(image)
        
        return image, label_code


def get_all_labels(imgdir, specified_dpi, specified_font_style):
    all_labels = set()
    for root, _, files in os.walk(imgdir):
        path_parts = root.split(os.sep)
        if len(path_parts) >= 4:
            char_code = path_parts[-3]  # Character code
            dpi = path_parts[-2]        # DPI
            font_style = path_parts[-1] # Font style

            if (specified_dpi == 'all' or dpi == specified_dpi) and (specified_font_style == 'all' or font_style == specified_font_style):
                for file in files:
                    if file.endswith(".bmp"): 
                        all_labels.add(int(char_code))
    return sorted(all_labels)


def split_dataset(train_Dataset, test_Dataset, ifsame, train_ratio = 0.8, val_ratio = 0.1):

    total_size = len(train_Dataset)
    train_size = int(total_size * train_ratio)
	
    if ifsame:
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # use random_split to split the dataset
        train_dataset, val_dataset, test_dataset = random_split(train_Dataset, [train_size, val_size, test_size])

    else:
        # for different training and test sets, subset samples directly from the test set
        val_size = total_size - train_size
        test_size = val_size
        train_dataset, val_dataset = random_split(train_Dataset, [train_size, val_size])
        test_dataset, _ = random_split(test_Dataset, [test_size, len(test_Dataset) - test_size])
    
    return train_dataset, val_dataset, test_dataset


def save_label_mapping(label_mapping, filepath='label_mapping.json'):
    with open(filepath, 'w') as f:
        json.dump(label_mapping, f)


def main():
    # Set parameters
    # directory_path = '/scratch/lt2326-2926-h24/ThaiOCR'
    # os.chdir(directory_path)

    ## 1. Parse command line arguments
    # create argument parser
    parser = argparse.ArgumentParser(description="Select the dataset configuration for training and testing")
    parser.add_argument('language', type=str, help="Specify the language (e.g., 'Thai', 'English', 'both')")
    parser.add_argument('train_dpi', type=str, help="Specify the DPI for training set(e.g., 200, 300, 400 or 'all')")
    parser.add_argument('train_font_style', type=str, help="Specify the font style for training set(e.g., 'normal', 'bold', or 'all')")
    parser.add_argument('test_dpi', type=str, help="Specify the DPI for test set(e.g., 200, 300, 400 or 'all')")
    parser.add_argument('test_font_style', type=str, help="Specify the font style for test set(e.g., 'normal', 'bold', or 'all')")
    args = parser.parse_args()

    # get specified arguments
    language = args.language
    train_dpi = args.train_dpi
    train_font_style = args.train_font_style
    test_dpi = args.test_dpi
    test_font_style = args.test_font_style

    # 2. Create dataset and dataloader
    trainingdir = "/scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/Thai"
    # trainingdir = "ThaiOCR-TrainigSet"
    # data pre-processing
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # resize all images to 64x64
        transforms.Grayscale(num_output_channels=1),  # ensure single-channel grayscale if needed
        transforms.ToTensor() # transform to tensors
    ])

    ## create a global label mapping
    train_labels = get_all_labels(trainingdir, train_dpi, train_font_style)
    test_labels = get_all_labels(trainingdir, test_dpi, test_font_style)

    # Combine the labels from both training and testing sets to ensure consistency
    all_labels = sorted(set(train_labels).union(set(test_labels)))
    label_mapping = {label: idx for idx, label in enumerate(all_labels)}
    save_label_mapping(label_mapping) # save for further calling
    # print("label_mapping:", label_mapping)

    # load data (prepare for further split)
    train_Dataset = ThaiCharDataset(trainingdir, language, train_dpi, train_font_style, label_mapping, transform=transform)
    test_Dataset = ThaiCharDataset(trainingdir, language, test_dpi, test_font_style, label_mapping, transform=transform)

    # consider two conditions: the same/different training and testing sets
    ifsame = (train_dpi == test_dpi) and (train_font_style == test_font_style)

    # split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(train_Dataset, test_Dataset, ifsame)

    # save
    torch.save(train_dataset, 'train_dataset.pth')
    torch.save(val_dataset, 'val_dataset.pth')  
    torch.save(test_dataset, 'test_dataset.pth')  

    print("The dataset has been successfully split into training, validation, and test sets!")

if __name__ == "__main__":
    main()
