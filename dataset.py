import pickle
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

class CIFAR10Raw(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with the 'data_batch_1', etc.
            train (bool): If True, loads data_batch_1 to 5. If False, loads test_batch.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.targets = []

        # 1. Define which files to load based on train/test mode
        if train:
            file_list = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            file_list = ['test_batch']

        # 2. Loop through files and unpickle
        for file_name in file_list:
            file_path = os.path.join(root_dir, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='bytes')
                
                # entry is a dict with keys: b'batch_label', b'labels', b'data', b'filenames'
                self.data.append(entry[b'data'])
                self.targets.extend(entry[b'labels'])

        # 3. Concatenate all data batches into one big array
        # Shape becomes (50000, 3072)
        self.data = np.vstack(self.data)

        # 4. Pre-reshape the data to simplify __getitem__
        # Original: (N, 3072) where 3072 = R(1024) + G(1024) + B(1024)
        # Reshape to (N, 3, 32, 32)
        self.data = self.data.reshape(-1, 3, 32, 32)
        
        # Convert to HWC format (N, 32, 32, 3) because PIL and many libraries expect channel last
        self.data = self.data.transpose((0, 2, 3, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. Grab the image and label
        img, target = self.data[idx], self.targets[idx]

        # 2. Convert raw numpy array to PIL Image
        # This is standard practice so we can use decent transforms (Flip, Rotate, etc.)
        img = Image.fromarray(img)

        # 3. Apply transforms (Normalization happens here)
        if self.transform:
            img = self.transform(img)

        # 4. Return tuple
        return img, target

def get_dataloader(root_dir, batch_size=64, num_workers=4, train=True):
    """
    Helper function to get the dataloader ready for diffusion training.
    """
    # Diffusion Specific Transforms
    transform = T.Compose([
        # T.RandomHorizontalFlip(), # Augmentation
        T.ToTensor(),             # Converts [0, 255] to [0.0, 1.0]
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Converts [0, 1] to [-1, 1]
    ])

    dataset = CIFAR10Raw(root_dir=root_dir, train=train, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,      # Shuffle only if training
        num_workers=num_workers,
        pin_memory=True,    # Faster transfer to CUDA
        drop_last=True      # Drop incomplete batches (helps with geometry calc)
    )
    
    return loader

# Simple test block to verify shapes
if __name__ == "__main__":
    # Point this to where you unzipped the cifar folder
    # e.g., root_dir = "./cifar-10-batches-py"
    print("Dataset file ready. Import 'get_dataloader' in train.py")