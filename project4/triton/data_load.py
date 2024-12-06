import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import struct

def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_images = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        return images

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(4)  # magic number
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def normalize(image):
    return image / 255.0

def read_dir(dir_path):
    train_images = read_mnist_images(dir_path + '/train-images-idx3-ubyte')
    train_labels = read_mnist_labels(dir_path + '/train-labels-idx1-ubyte')
    test_images = read_mnist_images(dir_path + '/t10k-images-idx3-ubyte')
    test_labels = read_mnist_labels(dir_path + '/t10k-labels-idx1-ubyte')
    return train_images, train_labels, test_images, test_labels
