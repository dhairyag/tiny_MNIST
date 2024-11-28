from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.swa_utils import AveragedModel, SWALR
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),  # 1*3*3*4 + 4 = 40 params
            nn.ReLU(),
            nn.BatchNorm2d(4),              # 8 params
            nn.Conv2d(4, 4, 3, padding=1),  # 4*3*3*4 + 4 = 148 params
            nn.ReLU(),
            nn.BatchNorm2d(4),              # 8 params
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),  # 4*3*3*8 + 8 = 296 params
            nn.ReLU(),
            nn.BatchNorm2d(8),              # 16 params
            nn.Conv2d(8, 8, 3, padding=1),  # 8*3*3*8 + 8 = 584 params
            nn.ReLU(),
            nn.BatchNorm2d(8),              # 16 params
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1), # 8*3*3*12 + 12 = 876 params
            nn.ReLU(),
            nn.BatchNorm2d(12),             # 24 params
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )
        
        # 12 channels * 3 * 3 = 108 neurons after three max pools (28->14->7->3)
        self.fc1 = nn.Linear(12 * 3 * 3, 10)  # 108*10 + 10 = 1090 params
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 12 * 3 * 3)  # Flatten
        x = F.dropout(x, p=0.1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

#!pip install torchinfo
from torchinfo import summary
use_cuda = torch.cuda.is_available()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if use_cuda else "cpu")
#model = Net().to(device)
# Create a dummy input tensor on the correct device
#summary(model, input_size=(1, 1, 28, 28), device=device)



torch.manual_seed(1456)
batch_size = 512

kwargs = {'num_workers': 4, 'pin_memory': True} if device.type in ["cuda", "mps"] else {}

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import random
# Define the augmentation pipeline
train_transforms = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=20,
        p=0.8,
        border_mode=cv2.BORDER_CONSTANT,
        value=0
    ),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.Perspective(scale=(0.05, 0.15), p=0.4, keep_size=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=0),

    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.Blur(blur_limit=3, p=0.2),
    
    A.ElasticTransform(
        alpha=1.2,
        sigma=12.0,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=0.4
    ),
    
    A.CoarseDropout(
        max_holes=3,
        max_height=10,
        max_width=10,
        min_holes=1,
        fill_value=0,
        p=0.3
    ),

    A.Normalize(
        mean=[0.1307],
        std=[0.3081],
    ),
    ToTensorV2(),
])

# Custom Dataset class to work with Albumentations
class MNISTAlbumentations(datasets.MNIST):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__(root, train=train, download=download, transform=None)
        self.transform = transform
        
    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        
        # Convert to numpy array and add channel dimension
        img = np.array(img)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension for Albumentations
        
        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]
            
        return img, label


# Test transforms (only normalization, no augmentation)
test_transforms = A.Compose([
    A.Normalize(
        mean=[0.1307],
        std=[0.3081],
    ),
    ToTensorV2(),
])


# Optional: Visualization function to check augmentations
def visualize_augmentations(dataset, idx=0, samples=5):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(20, 4))
    for i in range(samples):
        data = dataset[idx][0]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if data.shape[0] == 1:  # If channels first, move to last
            data = np.transpose(data, (1, 2, 0))
        plt.subplot(1, samples, i + 1)
        plt.imshow(data.squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# Uncomment to visualize augmentations
#visualize_augmentations(train_loader.dataset)

# # print number of samples in train and test dataset
#print(f"Number of samples in train dataset: {len(train_loader.dataset)}")
#print(f"Number of samples in test dataset: {len(test_loader.dataset)}")

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        pbar.set_description(desc= f'epoch={epoch} loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    # 1. Initial setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if use_cuda else "cpu")
    torch.manual_seed(1456)
    batch_size = 512
    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type in ["cuda", "mps"] else {}

    # 2. Create data loaders (do this only once)
    train_loader = torch.utils.data.DataLoader(
        MNISTAlbumentations('../data', train=True, download=True, transform=train_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MNISTAlbumentations('../data', train=False, transform=test_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)

    # 3. Create model and print summary once
    model = Net().to(device)
    summary(model, input_size=(1, 1, 28, 28), device=device)

    # 4. Setup optimizers and schedulers
    swa_model = AveragedModel(model)
    optimizer = optim.SGD(model.parameters(), 
                         lr=0.001,
                         momentum=0.9,
                         weight_decay=5e-4,
                         nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.25,
        epochs=25,
        steps_per_epoch=len(train_loader),
        pct_start=0.25,
        div_factor=30,
        final_div_factor=1e4
    )

    swa_start = 15
    swa_scheduler = SWALR(optimizer, swa_lr=0.0005)

    # 5. Training loop
    warmup_epochs = 2
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs * len(train_loader)
    )

    for epoch in range(1, 26):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        if epoch <= warmup_epochs:
            train(model, device, train_loader, optimizer, epoch, warmup_lr_scheduler)
        elif epoch < swa_start:
            train(model, device, train_loader, optimizer, epoch, scheduler)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.001
            train(model, device, train_loader, optimizer, epoch)
            swa_model.update_parameters(model)
        
        test(model, device, test_loader)
        
        if epoch >= swa_start:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            test(swa_model, device, test_loader)