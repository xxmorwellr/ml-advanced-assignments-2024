import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# import imagehash # pre-intall: pip install imagehash Pillow
# from PIL import Image


class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False
        # self.hash = None  # add hash attribute

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label, self.filename)).float()
            self.loaded = True
        
            # # hash
            # pil_image = Image.open(os.path.join(self.imgdir, self.label, self.filename)).convert("RGB")
            # self.hash = str(imagehash.phash(pil_image)) 

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        # hashes = set()  # add hash set to detect duplicates
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            # print("arttype:",arttype)
            artfiles = item[2]
            # print("artfiles:",artfiles)
            for art in artfiles:
                img = WikiArtImage(imgdir, arttype, art)
                img.get() 
                # # check if hash exists
                # if img.hash not in hashes:
                filedict[art] = img
                indices.append(art)
                classes.add(arttype)
                #     hashes.add(img.hash)  

        # print('Classes set:',classes) # unordered set => convert to list later
        missing_arttype = 'Action_painting'
        if missing_arttype not in classes:
            classes.add(missing_arttype)
        print("...finished")

        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
        self.classes = list(classes) 
        self.classes = sorted(classes) # ordered
        # print('Classes List:',self.classes)
        self.device = device

        # calculate class weight
        self.class_weights = self.calculate_class_weights(device)
        # print("class_weights:",self.class_weights)
    
    def calculate_class_weights(self, device):
        class_counts = [0] * len(self.classes)
        for art in self.filedict.values():
            label = art.label
            class_index = self.classes.index(label) # collect list indices
            class_counts[class_index] += 1
        
        min_weight = 1e-6
        alpha = 0.5 # smoothing 
        weights = [(1.0 / count)** alpha if count > 0 else min_weight for count in class_counts]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        return torch.FloatTensor(weights).to(device)
    
        
    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.classes.index(imgobj.label)
        image = imgobj.get().to(self.device)

        return image, ilabel, imgobj.label 

class WikiArtModel(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, (4,4), padding=2) # increase `out_channels`
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, (4,4), padding=2) # add a convolution layer
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d((4,4), padding=2)

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64 * 27 * 27, 128)  # according to image size
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.pool(self.relu(self.bn1(self.conv1(image))))
        output = self.pool(self.relu(self.bn2(self.conv2(output))))
        # print("Output shape before fully connected layer:", output.shape)
        output = output.view(output.size(0), -1) 
        output = self.dropout(self.relu(self.fc1(output)))   
        output = self.fc2(output)
        return self.softmax(output)
