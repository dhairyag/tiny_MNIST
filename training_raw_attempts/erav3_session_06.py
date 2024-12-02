from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.swa_utils import AveragedModel, SWALR
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
        
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(1, 4, 3, padding=1),  # 1*3*3*4 + 4 = 40 params
#             nn.ReLU(),
#             nn.BatchNorm2d(4),              # 8 params
#             nn.Conv2d(4, 4, 3, padding=1),  # 4*3*3*4 + 4 = 148 params
#             nn.ReLU(),
#             nn.BatchNorm2d(4),              # 8 params
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.05)
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(4, 8, 3, padding=1),  # 4*3*3*8 + 8 = 296 params
#             nn.ReLU(),
#             nn.BatchNorm2d(8),              # 16 params
#             nn.Conv2d(8, 8, 3, padding=1),  # 8*3*3*8 + 8 = 584 params
#             nn.ReLU(),
#             nn.BatchNorm2d(8),              # 16 params
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.05)
#         )
        
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(8, 12, 3, padding=1), # 8*3*3*12 + 12 = 876 params
#             nn.ReLU(),
#             nn.BatchNorm2d(12),             # 24 params
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.05)
#         )
        
#         # 12 channels * 3 * 3 = 108 neurons after three max pools (28->14->7->3)
#         self.fc1 = nn.Linear(12 * 3 * 3, 10)  # 108*10 + 10 = 1090 params
        
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = x.view(-1, 12 * 3 * 3)  # Flatten
#         x = F.dropout(x, p=0.0)
#         x = self.fc1(x)
#         return F.log_softmax(x, dim=1)
# 4. # Squeeze-and-Excitation Block Definition
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# SE-Enhanced Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            SEBlock(4),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            SEBlock(4),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            SEBlock(8),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            SEBlock(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            SEBlock(12),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05)
        )
        
        self.fc1 = nn.Linear(12 * 3 * 3, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 12 * 3 * 3)
        #x = F.dropout(x, p=0.0)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

#!pip install torchinfo
from torchinfo import summary
use_cuda = torch.cuda.is_available()
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if use_cuda else "cpu")
#model = Net().to(device)
# Create a dummy input tensor on the correct device
#summary(model, input_size=(1, 1, 28, 28), device=device)




#kwargs = {'num_workers': 4, 'pin_memory': True} if device.type in ["cuda", "mps"] else {}

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import random
# Define the augmentation pipeline
train_transforms = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.05,
        rotate_limit=16,
        p=0.15,
        border_mode=cv2.BORDER_CONSTANT,
        value=0
    ),
    # A.ShiftScaleRotate(
    #     shift_limit=0.0625,
    #     scale_limit=0.1,
    #     rotate_limit=15,
    #     p=0.7,
    #     border_mode=cv2.BORDER_CONSTANT,
    #     value=0
    # ),
    # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    # A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
    # A.Perspective(scale=(0.05, 0.1), p=0.3, keep_size=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=0),

    #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    #A.Blur(blur_limit=3, p=0.2),
    
    # A.ElasticTransform(
    #     alpha=1.0,
    #     sigma=10.0,
    #     interpolation=cv2.INTER_LINEAR,
    #     border_mode=cv2.BORDER_CONSTANT,
    #     value=0,
    #     p=0.3
    # ),
    
    # A.CoarseDropout(
    #     max_holes=3,
    #     max_height=8,
    #     max_width=8,
    #     min_holes=1,
    #     fill_value=0,
    #     p=0.2
    # ),

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
    kwargs = {'num_workers': 4, 'pin_memory': True} if device.type in ["cuda", "mps"] else {}

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
    # swa_model = AveragedModel(model)
    # optimizer = optim.SGD(model.parameters(), 
    #                      lr=0.001,
    #                      momentum=0.9,
    #                      weight_decay=5e-4,
    #                      nesterov=True)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.25,
    #     epochs=15,
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.3,
    #     div_factor=25,
    #     final_div_factor=1e4
    # )

    # swa_start = 6
    # swa_scheduler = SWALR(optimizer, swa_lr=0.0005)

    # # 5. Training loop
    # warmup_epochs = 2
    # warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, 
    #     start_factor=0.1,
    #     end_factor=1.0,
    #     total_iters=warmup_epochs * len(train_loader)
    # )

    # for epoch in range(1, 15):
    #     current_lr = optimizer.param_groups[0]['lr']
    #     print(f'Current learning rate: {current_lr:.6f}')
        
    #     if epoch <= warmup_epochs:
    #         train(model, device, train_loader, optimizer, epoch, warmup_lr_scheduler)
    #     elif epoch < swa_start:
    #         train(model, device, train_loader, optimizer, epoch, scheduler)
    #     else:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = 0.001
    #         train(model, device, train_loader, optimizer, epoch)
    #         swa_model.update_parameters(model)
        
    #     test(model, device, test_loader)
        
    #     if epoch >= swa_start:
    #         torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    #         test(swa_model, device, test_loader)
    

    # model = Net().to(device)

    # swa_model = AveragedModel(model)

    # # Use OneCycleLR scheduler for better convergence
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=0.4,
    #     epochs=15,                          # Increased epochs
    #     steps_per_epoch=len(train_loader),
    #     pct_start=0.3,                      # Warm up for 30% of training
    #     div_factor=10,                      # Initial lr = max_lr/10
    #     final_div_factor=100                # Final lr = initial_lr/100
    # )

    # # Start SWA later in training
    # swa_start = 5
    # swa_scheduler = SWALR(optimizer, swa_lr=0.001)

    # for epoch in range(1, 15):  # Increased to 20 epochs
    #     # Get current learning rate
    #     current_lr = optimizer.param_groups[0]['lr']
    #     print(f'Current learning rate: {current_lr:.6f}')
        
    #     if epoch < swa_start:
    #         train(model, device, train_loader, optimizer, epoch, scheduler)
    #     else:
    #         train(model, device, train_loader, optimizer, epoch)
    #         swa_model.update_parameters(model)
    #         swa_scheduler.step()
        
    #     test(model, device, test_loader)
        
    #     if epoch >= swa_start:
    #         torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    #         test(swa_model, device, test_loader)
    
    # --------------------------------------------------------------------------------
    # Tunable parameters
    # max_lr = 0.3          # Peak learning rate
    # initial_div = 25      # Divisor for initial learning rate (max_lr / initial_div)
    # final_div = 100       # Divisor for final learning rate
    # warmup_pct = 0.3      # Percentage of training for warmup
    # momentum = 0.9        # Momentum for SGD
    # weight_decay = 1e-4   # L2 regularization
    # num_epochs = 15       # Total number of epochs

    # #max_lr,	initial_div,	final_div,	warmup_pct,	momentum,	weight_decay
    # # 0.3,	25,	100,	0.3,	0.9,	1e-4
    # # 0.2,	20,	50,	0.4,	0.95,	1e-5
    # # 0.1,	10,	25,	0.2,	0.85,	5e-5
    # # 0.4,	30,	200,	0.5,	0.9,	1e-4
    # # 0.5,	50,	300,	0.1,	0.95,	1e-3
    # # Define parameter combinations
    # param_combinations = [
    #     # max_lr, initial_div, final_div, warmup_pct, momentum, weight_decay
    #     # (0.3, 25, 100, 0.3, 0.9, 1e-4),
    #     # (0.2, 20, 50, 0.4, 0.95, 1e-5),
    #     # (0.1, 10, 25, 0.2, 0.85, 5e-5),
    #     # (0.4, 30, 200, 0.5, 0.9, 1e-4),
    #     # (0.5, 50, 300, 0.1, 0.95, 1e-3),
    #     # # Additional combinations
    #     (0.3, 20, 50, 0.3, 0.9, 1e-5), # Best performing parameters - 99.33%
    #     # (0.3, 10, 25, 0.3, 0.9, 5e-5),
    #     # (0.3, 30, 200, 0.3, 0.9, 1e-4),
    #     # (0.2, 25, 100, 0.4, 0.95, 1e-4),
    #     # (0.2, 10, 25, 0.4, 0.95, 5e-5),
    #     # (0.2, 30, 200, 0.4, 0.95, 1e-4),
    #     # (0.1, 25, 100, 0.2, 0.85, 1e-4),
    #     # (0.1, 20, 50, 0.2, 0.85, 1e-5),
    #     # (0.1, 30, 200, 0.2, 0.85, 1e-4),
    #     (0.4, 25, 100, 0.5, 0.9, 1e-4), # Best performing parameters - 99.32%
    #     # (0.4, 20, 50, 0.5, 0.9, 1e-5),
    #     # (0.4, 10, 25, 0.5, 0.9, 5e-5),
    #     # (0.5, 25, 100, 0.1, 0.95, 1e-4),
    #     # (0.5, 20, 50, 0.1, 0.95, 1e-5),
    #     # (0.5, 10, 25, 0.1, 0.95, 5e-5),
    #     # New combinations exploring around best performing parameters
    #     # (0.35, 20, 50, 0.3, 0.9, 1e-5),  # Between 0.3 and 0.4 max_lr
    #     # (0.35, 25, 100, 0.4, 0.9, 1e-5),  # Testing different warmup
    #     # (0.3, 22, 75, 0.3, 0.9, 1e-5),    # Fine-tuning div factors
    #     # (0.4, 22, 75, 0.4, 0.9, 1e-5),    # Fine-tuning div factors
    #     # (0.35, 20, 50, 0.35, 0.92, 1e-5), # Testing intermediate momentum
    #     (0.3, 20, 50, 0.25, 0.9, 8e-6),   # Lower warmup, finer weight decay : Best performing parameters - 99.31%  
    #     # (0.4, 25, 75, 0.45, 0.9, 8e-6),   # Balanced div factors
    #     # (0.35, 18, 45, 0.35, 0.9, 1e-5),  # Lower div factors
    #     # (0.32, 20, 50, 0.3, 0.9, 1e-5),   # Fine-tuning max_lr
    #     # (0.38, 22, 60, 0.4, 0.9, 1e-5),   # Balanced parameters
    # ]

    # # Store results
    # results = []
    # num_epochs = 15

    # # Run training for each parameter combination
    # for params in param_combinations:
    #     max_lr, initial_div, final_div, warmup_pct, momentum, weight_decay = params
    #     print(f"\nTraining with parameters:")
    #     print(f"max_lr: {max_lr}, initial_div: {initial_div}, final_div: {final_div}")
    #     print(f"warmup_pct: {warmup_pct}, momentum: {momentum}, weight_decay: {weight_decay}")

    #     # Initialize model
    #     model = Net().to(device)
        
    #     # Optimizer
    #     optimizer = optim.SGD(
    #         model.parameters(),
    #         lr=max_lr / initial_div,
    #         momentum=momentum,
    #         weight_decay=weight_decay
    #     )

    #     # Scheduler
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=max_lr,
    #         epochs=num_epochs,
    #         steps_per_epoch=len(train_loader),
    #         pct_start=warmup_pct,
    #         div_factor=initial_div,
    #         final_div_factor=final_div,
    #     )

    #     # Track best accuracy for this combination
    #     best_accuracy = 0.0

    #     # Training Loop
    #     for epoch in range(1, num_epochs + 1):
    #         current_lr = optimizer.param_groups[0]['lr']
    #         print(f'Epoch {epoch}: Current learning rate: {current_lr:.6f}')
            
    #         train(model, device, train_loader, optimizer, epoch, scheduler)
            
    #         # Modified test function to return accuracy
    #         model.eval()
    #         correct = 0
    #         with torch.no_grad():
    #             for data, target in test_loader:
    #                 data, target = data.to(device), target.to(device)
    #                 output = model(data)
    #                 pred = output.argmax(dim=1, keepdim=True)
    #                 correct += pred.eq(target.view_as(pred)).sum().item()
            
    #         accuracy = 100. * correct / len(test_loader.dataset)
    #         print(f'\nTest Accuracy: {accuracy:.2f}%\n')
            
    #         best_accuracy = max(best_accuracy, accuracy)

    #     # Store results for this combination
    #     results.append({
    #         'params': params,
    #         'best_accuracy': best_accuracy
    #     })

    # # Print and plot final results
    # print("\nFinal Results:")
    # print("-" * 80)
    # for result in results:
    #     params = result['params']
    #     print(f"Parameters: max_lr={params[0]}, initial_div={params[1]}, final_div={params[2]}, "
    #           f"warmup_pct={params[3]}, momentum={params[4]}, weight_decay={params[5]}")
    #     print(f"Best Accuracy: {result['best_accuracy']:.2f}%")
    #     print("-" * 80)

    # # Plot results
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))
    # accuracies = [r['best_accuracy'] for r in results]
    # param_labels = [f"Combo {i+1}" for i in range(len(results))]
    
    # plt.bar(param_labels, accuracies)
    # plt.title('Best Test Accuracy for Different Parameter Combinations')
    # plt.xlabel('Parameter Combinations')
    # plt.ylabel('Best Test Accuracy (%)')
    # plt.ylim(min(accuracies) - 1, max(accuracies) + 1)
    
    # # Add value labels on top of each bar
    # for i, v in enumerate(accuracies):
    #     plt.text(i, v + 0.1, f'{v:.2f}%', ha='center')
    
    # plt.tight_layout()
    # plt.show()

    # Final Results:
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=25, final_div=100, warmup_pct=0.3, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.12%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.2, initial_div=20, final_div=50, warmup_pct=0.4, momentum=0.95, weight_decay=1e-05
    # Best Accuracy: 99.24%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.1, initial_div=10, final_div=25, warmup_pct=0.2, momentum=0.85, weight_decay=5e-05
    # Best Accuracy: 99.14%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.19%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.5, initial_div=50, final_div=300, warmup_pct=0.1, momentum=0.95, weight_decay=0.001
    # Best Accuracy: 99.26%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.33%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=10, final_div=25, warmup_pct=0.3, momentum=0.9, weight_decay=5e-05
    # Best Accuracy: 99.13%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=30, final_div=200, warmup_pct=0.3, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.24%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.2, initial_div=25, final_div=100, warmup_pct=0.4, momentum=0.95, weight_decay=0.0001
    # Best Accuracy: 99.22%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.2, initial_div=10, final_div=25, warmup_pct=0.4, momentum=0.95, weight_decay=5e-05
    # Best Accuracy: 99.13%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.2, initial_div=30, final_div=200, warmup_pct=0.4, momentum=0.95, weight_decay=0.0001
    # Best Accuracy: 99.09%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.1, initial_div=25, final_div=100, warmup_pct=0.2, momentum=0.85, weight_decay=0.0001
    # Best Accuracy: 99.20%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.1, initial_div=20, final_div=50, warmup_pct=0.2, momentum=0.85, weight_decay=1e-05
    # Best Accuracy: 99.09%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.1, initial_div=30, final_div=200, warmup_pct=0.2, momentum=0.85, weight_decay=0.0001
    # Best Accuracy: 99.10%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=100, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.32%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=20, final_div=50, warmup_pct=0.5, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.12%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=10, final_div=25, warmup_pct=0.5, momentum=0.9, weight_decay=5e-05
    # Best Accuracy: 99.09%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.5, initial_div=25, final_div=100, warmup_pct=0.1, momentum=0.95, weight_decay=0.0001
    # Best Accuracy: 99.12%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.5, initial_div=20, final_div=50, warmup_pct=0.1, momentum=0.95, weight_decay=1e-05
    # Best Accuracy: 99.14%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.5, initial_div=10, final_div=25, warmup_pct=0.1, momentum=0.95, weight_decay=5e-05
    # Best Accuracy: 99.13%
    # --------------------------------------------------------------------------------

    # 2nd round of experiments

    # Final Results:
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.35, initial_div=20, final_div=50, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.09%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.35, initial_div=25, final_div=100, warmup_pct=0.4, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.28%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=22, final_div=75, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.23%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=22, final_div=75, warmup_pct=0.4, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.08%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.35, initial_div=20, final_div=50, warmup_pct=0.35, momentum=0.92, weight_decay=1e-05
    # Best Accuracy: 99.25%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.25, momentum=0.9, weight_decay=8e-06
    # Best Accuracy: 99.31%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=75, warmup_pct=0.45, momentum=0.9, weight_decay=8e-06
    # Best Accuracy: 99.28%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.35, initial_div=18, final_div=45, warmup_pct=0.35, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.15%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.32, initial_div=20, final_div=50, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.18%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.38, initial_div=22, final_div=60, warmup_pct=0.4, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.22%
    
    # ----------- with Image augmentation -------------------------------------------------
    # Final Results:
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.38%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=100, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.29%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.25, momentum=0.9, weight_decay=8e-06
    # Best Accuracy: 99.21%
    # --------------------Final Results: A.ShiftScaleRotate(
    #     shift_limit=0.02,
    #     scale_limit=0.05,
    #     rotate_limit=10,
    #     p=0.2,
    #     border_mode=cv2.BORDER_CONSTANT,
    #     value=0
    # ),
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.26%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=100, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.37%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.25, momentum=0.9, weight_decay=8e-06
    # Best Accuracy: 99.22%
    # --------------------------------------------------------------------------------

    # # Define parameter combinations to test
    # param_combinations = [
    #     # max_lr, epochs, swa_start, momentum, weight_decay, swa_lr
    #     # (0.4, 15, 5, 0.9, 1e-4, 0.001),  # Base configuration
    #     # (0.3, 15, 7, 0.9, 1e-4, 0.001),  # Later SWA start
    #     # (0.5, 15, 3, 0.9, 1e-4, 0.001),  # Earlier SWA start
    #     # (0.4, 15, 5, 0.95, 1e-4, 0.0005), # Higher momentum, lower SWA lr
    #     # (0.3, 20, 8, 0.9, 5e-5, 0.001),  # More epochs, lower weight decay
    #     # (0.4, 12, 4, 0.85, 1e-4, 0.002),  # Fewer epochs, higher SWA lr
    #     # (0.2, 15, 5, 0.9, 1e-4, 0.0005),  # Lower max_lr
    #     # (0.15, 20, 8, 0.92, 8e-5, 0.0003), # Even lower max_lr, more epochs
    #     # (0.1, 20, 10, 0.95, 5e-5, 0.0002), # Very low max_lr, longest training
    #     # (0.08, 20, 7, 0.9, 1e-4, 0.0001)   # Lowest max_lr configuration
    #     (0.4, 15, 5, 0.9, 1e-4, 0.001),    # Best performing: 99.41% regular, 99.36% SWA -
    #     # (0.4, 13, 5, 0.9, 1e-4, 0.001),    # Best performing: 
    #     # (0.4, 11, 5, 0.9, 1e-4, 0.001),    # Best performing: 
    #     # (0.4, 18, 5, 0.9, 1e-4, 0.001),    # Best performing: more epochs
    #     # (0.4, 15, 4, 0.9, 1e-4, 0.001),    # Try slightly earlier SWA start
    #     # (0.4, 15, 6, 0.9, 1e-4, 0.001),    # Try slightly later SWA start
    #     # (0.4, 17, 5, 0.9, 1e-4, 0.001),    # Try few more epochs
    #     # (0.4, 15, 5, 0.9, 8e-5, 0.001),    # Try slightly lower weight decay
    #     # (0.4, 15, 5, 0.92, 1e-4, 0.001),   # Try slightly higher momentum
    #     # (0.35, 15, 5, 0.9, 1e-4, 0.001),   # Try slightly lower max_lr
    #     # (0.45, 15, 5, 0.9, 1e-4, 0.001),   # Try slightly higher max_lr
    #     (0.4, 15, 5, 0.9, 1e-4, 0.0008),   # Try slightly lower swa_lr -
    #     # (0.4, 15, 5, 0.9, 1e-4, 0.0012)    # Try slightly higher swa_lr
    # ]

    # # Store results for comparison
    # results = []

    # for params in param_combinations:
    #     max_lr, num_epochs, swa_start, momentum, weight_decay, swa_lr = params
    #     print(f"\nTraining with parameters:")
    #     print(f"max_lr: {max_lr}, epochs: {num_epochs}, swa_start: {swa_start}")
    #     print(f"momentum: {momentum}, weight_decay: {weight_decay}, swa_lr: {swa_lr}")

    #     # Initialize model and move to device
    #     model = Net().to(device)
    #     swa_model = AveragedModel(model)

    #     # Configure optimizer
    #     optimizer = optim.SGD(model.parameters(), lr=max_lr/10, momentum=momentum, weight_decay=weight_decay)

    #     # OneCycleLR scheduler
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=max_lr,
    #         epochs=num_epochs,
    #         steps_per_epoch=len(train_loader),
    #         pct_start=0.3,
    #         div_factor=10,
    #         final_div_factor=100
    #     )

    #     swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

    #     # Track best accuracy for this combination
    #     best_accuracy = 0.0
    #     best_swa_accuracy = 0.0

    #     for epoch in range(1, num_epochs + 1):
    #         current_lr = optimizer.param_groups[0]['lr']
    #         print(f'Epoch {epoch}: Current learning rate: {current_lr:.6f}')
            
    #         if epoch < swa_start:
    #             train(model, device, train_loader, optimizer, epoch, scheduler)
    #         else:
    #             train(model, device, train_loader, optimizer, epoch)
    #             swa_model.update_parameters(model)
    #             swa_scheduler.step()
            
    #         # Test regular model
    #         model.eval()
    #         correct = 0
    #         with torch.no_grad():
    #             for data, target in test_loader:
    #                 data, target = data.to(device), target.to(device)
    #                 output = model(data)
    #                 pred = output.argmax(dim=1, keepdim=True)
    #                 correct += pred.eq(target.view_as(pred)).sum().item()
            
    #         accuracy = 100. * correct / len(test_loader.dataset)
    #         best_accuracy = max(best_accuracy, accuracy)
    #         print(f'Regular Model Test Accuracy: {accuracy:.2f}%')
            
    #         # Test SWA model if applicable
    #         if epoch >= swa_start:
    #             torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    #             swa_model.eval()
    #             correct = 0
    #             with torch.no_grad():
    #                 for data, target in test_loader:
    #                     data, target = data.to(device), target.to(device)
    #                     output = swa_model(data)
    #                     pred = output.argmax(dim=1, keepdim=True)
    #                     correct += pred.eq(target.view_as(pred)).sum().item()
                
    #             swa_accuracy = 100. * correct / len(test_loader.dataset)
    #             best_swa_accuracy = max(best_swa_accuracy, swa_accuracy)
    #             print(f'SWA Model Test Accuracy: {swa_accuracy:.2f}%')

    #     # Store results
    #     results.append({
    #         'params': params,
    #         'best_regular_accuracy': best_accuracy,
    #         'best_swa_accuracy': best_swa_accuracy
    #     })

    # # Print final results
    # print("\nFinal Results:")
    # print("-" * 80)
    # for result in results:
    #     params = result['params']
    #     print(f"Parameters: max_lr={params[0]}, epochs={params[1]}, swa_start={params[2]}, "
    #           f"momentum={params[3]}, weight_decay={params[4]}, swa_lr={params[5]}")
    #     print(f"Best Regular Accuracy: {result['best_regular_accuracy']:.2f}%")
    #     print(f"Best SWA Accuracy: {result['best_swa_accuracy']:.2f}%")
    #     print("-" * 80)

    # # Plot comparison
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))
    # x = range(len(results))
    # regular_accuracies = [r['best_regular_accuracy'] for r in results]
    # swa_accuracies = [r['best_swa_accuracy'] for r in results]

    # plt.plot(x, regular_accuracies, 'b-', label='Regular Model')
    # plt.plot(x, swa_accuracies, 'r-', label='SWA Model')
    # plt.xlabel('Parameter Combination')
    # plt.ylabel('Best Accuracy (%)')
    # plt.title('Model Performance Comparison')
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(x)
    # plt.show()
    #Final Results:
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.41%
    # Best SWA Accuracy: 99.36%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, epochs=15, swa_start=7, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.29%
    # Best SWA Accuracy: 99.23%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.5, epochs=15, swa_start=3, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.08%
    # Best SWA Accuracy: 99.04%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=5, momentum=0.95, weight_decay=0.0001, swa_lr=0.0005
    # Best Regular Accuracy: 99.19%
    # Best SWA Accuracy: 99.16%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, epochs=20, swa_start=8, momentum=0.9, weight_decay=5e-05, swa_lr=0.001
    # Best Regular Accuracy: 99.14%
    # Best SWA Accuracy: 99.16%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=12, swa_start=4, momentum=0.85, weight_decay=0.0001, swa_lr=0.002
    # Best Regular Accuracy: 99.04%
    # Best SWA Accuracy: 99.08%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.2, epochs=15, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.0005
    # Best Regular Accuracy: 99.24%
    # Best SWA Accuracy: 99.16%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.15, epochs=20, swa_start=8, momentum=0.92, weight_decay=8e-05, swa_lr=0.0003
    # Best Regular Accuracy: 99.28%
    # Best SWA Accuracy: 99.21%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.1, epochs=20, swa_start=10, momentum=0.95, weight_decay=5e-05, swa_lr=0.0002
    # Best Regular Accuracy: 99.10%
    # Best SWA Accuracy: 99.11%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.08, epochs=20, swa_start=7, momentum=0.9, weight_decay=0.0001, swa_lr=0.0001
    # Best Regular Accuracy: 99.18%
    # Best SWA Accuracy: 99.17%
    # --------------------------------------------------------------------------------
    
    # 3rd round of experiments
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.41%
    # Best SWA Accuracy: 99.36%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=13, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.17%
    # Best SWA Accuracy: 99.13%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=11, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.00%
    # Best SWA Accuracy: 99.01%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=18, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.18%
    # Best SWA Accuracy: 99.08%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=4, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.18%
    # Best SWA Accuracy: 99.16%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=6, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.26%
    # Best SWA Accuracy: 99.24%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=17, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.26%
    # Best SWA Accuracy: 99.23%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=5, momentum=0.9, weight_decay=8e-05, swa_lr=0.001
    # Best Regular Accuracy: 99.04%
    # Best SWA Accuracy: 99.05%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=5, momentum=0.92, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.12%
    # Best SWA Accuracy: 99.06%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.35, epochs=15, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.20%
    # Best SWA Accuracy: 99.06%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.45, epochs=15, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.001
    # Best Regular Accuracy: 99.19%
    # Best SWA Accuracy: 99.08%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.0008
    # Best Regular Accuracy: 99.29%
    # Best SWA Accuracy: 99.16%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, epochs=15, swa_start=5, momentum=0.9, weight_decay=0.0001, swa_lr=0.0012
    # Best Regular Accuracy: 99.21%
    # Best SWA Accuracy: 99.11%
    # --------------------------------------------------------------------------------

    # # Tunable parameters for Adam
    # learning_rate = 0.001  # Initial learning rate
    # beta1 = 0.9           # First momentum coefficient
    # beta2 = 0.999         # Second momentum coefficient
    # eps = 1e-8            # Small constant for numerical stability
    # weight_decay = 1e-4   # L2 regularization
    # num_epochs = 15       # Total number of epochs

    # # Define parameter combinations for Adam
    # param_combinations = [
    #     # lr, beta1, beta2, eps, weight_decay
    #     (0.001, 0.9, 0.999, 1e-8, 1e-4),  # Default Adam parameters
    #     (0.003, 0.9, 0.999, 1e-8, 1e-4),  # Higher learning rate
    #     (0.0003, 0.9, 0.999, 1e-8, 1e-4), # Lower learning rate
    #     (0.001, 0.95, 0.999, 1e-8, 1e-4), # Higher beta1
    #     (0.001, 0.9, 0.9995, 1e-8, 1e-4), # Higher beta2
    #     (0.001, 0.9, 0.999, 1e-8, 1e-5),  # Lower weight decay
    #     (0.005, 0.9, 0.999, 1e-8, 1e-4),  # Even higher learning rate
    #     (0.008, 0.9, 0.999, 1e-8, 1e-4),  # Much higher learning rate
    #     (0.01, 0.9, 0.999, 1e-8, 1e-4),   # Highest learning rate
    # ]

    # # Store results
    # results = []
    # num_epochs = 15

    # # Run training for each parameter combination
    # for params in param_combinations:
    #     lr, beta1, beta2, eps, weight_decay = params
    #     print(f"\nTraining with parameters:")
    #     print(f"learning_rate: {lr}, beta1: {beta1}, beta2: {beta2}")
    #     print(f"eps: {eps}, weight_decay: {weight_decay}")

    #     # Initialize model
    #     model = Net().to(device)
        
    #     # Adam Optimizer
    #     optimizer = optim.Adam(
    #         model.parameters(),
    #         lr=lr,
    #         betas=(beta1, beta2),
    #         eps=eps,
    #         weight_decay=weight_decay
    #     )

    #     # Track best accuracy for this combination
    #     best_accuracy = 0.0

    #     # Training Loop
    #     for epoch in range(1, num_epochs + 1):
    #         print(f'Epoch {epoch}')
            
    #         train(model, device, train_loader, optimizer, epoch)
            
    #         # Modified test function to return accuracy
    #         model.eval()
    #         correct = 0
    #         with torch.no_grad():
    #             for data, target in test_loader:
    #                 data, target = data.to(device), target.to(device)
    #                 output = model(data)
    #                 pred = output.argmax(dim=1, keepdim=True)
    #                 correct += pred.eq(target.view_as(pred)).sum().item()
            
    #         accuracy = 100. * correct / len(test_loader.dataset)
    #         print(f'\nTest Accuracy: {accuracy:.2f}%\n')
            
    #         best_accuracy = max(best_accuracy, accuracy)

    #     # Store results for this combination
    #     results.append({
    #         'params': params,
    #         'best_accuracy': best_accuracy
    #     })

    # # Print and plot final results
    # print("\nFinal Results:")
    # print("-" * 80)
    # for result in results:
    #     params = result['params']
    #     print(f"Parameters: lr={params[0]}, beta1={params[1]}, beta2={params[2]}, "
    #           f"eps={params[3]}, weight_decay={params[4]}")
    #     print(f"Best Accuracy: {result['best_accuracy']:.2f}%")
    #     print("-" * 80)

    # # Plot results
    # import matplotlib.pyplot as plt

    # plt.figure(figsize=(12, 6))
    # accuracies = [r['best_accuracy'] for r in results]
    # param_labels = [f"Combo {i+1}" for i in range(len(results))]
    
    # plt.bar(param_labels, accuracies)
    # plt.title('Best Test Accuracy for Different Adam Parameter Combinations')
    # plt.xlabel('Parameter Combinations')
    # plt.ylabel('Best Test Accuracy (%)')
    # plt.ylim(min(accuracies) - 1, max(accuracies) + 1)
    
    # # Add value labels on top of each bar
    # for i, v in enumerate(accuracies):
    #     plt.text(i, v + 0.1, f'{v:.2f}%', ha='center')
    
    # plt.tight_layout()
    # plt.show()
    #     Final Results: with 15 epochs Adam
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 98.88%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.003, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 99.06%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.0003, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 97.97%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.001, beta1=0.95, beta2=0.999, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 98.74%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.001, beta1=0.9, beta2=0.9995, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 98.84%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.001, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=1e-05
    # Best Accuracy: 99.12%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.005, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 99.05%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.008, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 99.01%
    # --------------------------------------------------------------------------------
    # Parameters: lr=0.01, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=0.0001
    # Best Accuracy: 99.01%
    # --------------------------------------------------------------------------------

       # Tunable parameters for OneCycle policy
    max_lr = 0.4          # Peak learning rate
    initial_div = 20      # Divisor for initial learning rate (max_lr / initial_div)
    final_div = 50        # Divisor for final learning rate
    warmup_pct = 0.3      # Percentage of training for warmup
    momentum = 0.9        # Momentum for SGD
    weight_decay = 1e-5   # L2 regularization
    num_epochs = 15       # Total number of epochs

    # Define parameter combinations for OneCycle
    param_combinations = [
        # max_lr, initial_div, final_div, warmup_pct, momentum, weight_decay
        (0.3, 20, 50, 0.3, 0.9, 1e-5),  # Best performing parameters - 99.33% -e20
        #(0.4, 25, 100, 0.5, 0.9, 1e-4), # Second best - 99.32%
        #(0.3, 25, 100, 0.3, 0.9, 1e-4), # Baseline
        (0.2, 20, 50, 0.4, 0.95, 1e-5), # Lower learning rate -e20
        #(0.1, 10, 25, 0.2, 0.85, 5e-5), # Very low learning rate
        # (0.4, 30, 200, 0.5, 0.9, 1e-4), # Higher divs
        #(0.5, 50, 300, 0.1, 0.95, 1e-3), # Extreme values
        #(0.4, 30, 200, 0.5, 0.9, 1e-4),  # Best performing baseline - 99.38%
        (0.4, 25, 175, 0.5, 0.9, 1e-4),  # Slightly lower divs -e20
        # (0.4, 35, 225, 0.5, 0.9, 1e-4),  # Slightly higher divs
        # (0.35, 30, 200, 0.5, 0.9, 1e-4), # Lower max_lr
        (0.45, 30, 200, 0.5, 0.9, 1e-4), # Higher max_lr -e20
        # (0.4, 30, 200, 0.4, 0.9, 1e-4),  # Lower warmup
        # (0.4, 30, 200, 0.6, 0.9, 1e-4),  # Higher warmup
        # (0.4, 30, 200, 0.5, 0.85, 1e-4), # Lower momentum
        # (0.4, 30, 200, 0.5, 0.95, 1e-4), # Higher momentum
        # (0.4, 30, 200, 0.5, 0.9, 5e-5),  # Lower weight decay
        # (0.4, 30, 200, 0.5, 0.9, 2e-4),  # Higher weight decay
    ]

    # Store results
    results = []
    num_epochs = 20

    # Run training for each parameter combination
    for params in param_combinations:
        max_lr, initial_div, final_div, warmup_pct, momentum, weight_decay = params
        print(f"\nTraining with parameters:")
        print(f"max_lr: {max_lr}, initial_div: {initial_div}, final_div: {final_div}")
        print(f"warmup_pct: {warmup_pct}, momentum: {momentum}, weight_decay: {weight_decay}")

        # Initialize model
        model = Net().to(device)
        
        # SGD Optimizer with OneCycle policy
        optimizer = optim.SGD(
            model.parameters(),
            lr=max_lr/initial_div,  # Initial learning rate
            momentum=momentum,
            weight_decay=weight_decay
        )

        # OneCycle scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=warmup_pct,
            div_factor=initial_div,
            final_div_factor=final_div/initial_div
        )

        # Track best accuracy for this combination
        best_accuracy = 0.0

        # Training Loop
        for epoch in range(1, num_epochs + 1):
            print(f'Epoch {epoch}')
            
            train(model, device, train_loader, optimizer, epoch, scheduler)
            
            # Modified test function to return accuracy
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            accuracy = 100. * correct / len(test_loader.dataset)
            print(f'\nTest Accuracy: {accuracy:.2f}%\n')
            
            best_accuracy = max(best_accuracy, accuracy)

        # Store results for this combination
        results.append({
            'params': params,
            'best_accuracy': best_accuracy
        })

    # Print and plot final results
    print("\nFinal Results:")
    print("-" * 80)
    for result in results:
        params = result['params']
        print(f"Parameters: max_lr={params[0]}, initial_div={params[1]}, final_div={params[2]}, "
              f"warmup_pct={params[3]}, momentum={params[4]}, weight_decay={params[5]}")
        print(f"Best Accuracy: {result['best_accuracy']:.2f}%")
        print("-" * 80)

    # Plot results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    accuracies = [r['best_accuracy'] for r in results]
    param_labels = [f"Combo {i+1}" for i in range(len(results))]
    
    plt.bar(param_labels, accuracies)
    plt.title('Best Test Accuracy for Different OneCycle Parameter Combinations')
    plt.xlabel('Parameter Combinations')
    plt.ylabel('Best Test Accuracy (%)')
    plt.ylim(min(accuracies) - 1, max(accuracies) + 1)
    
    # Add value labels on top of each bar
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.1, f'{v:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.show()

    # Final Results:
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.26%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=100, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.15%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=25, final_div=100, warmup_pct=0.3, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.25%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.2, initial_div=20, final_div=50, warmup_pct=0.4, momentum=0.95, weight_decay=1e-05
    # Best Accuracy: 99.07%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.1, initial_div=10, final_div=25, warmup_pct=0.2, momentum=0.85, weight_decay=5e-05
    # Best Accuracy: 99.08%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.38% ****
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.5, initial_div=50, final_div=300, warmup_pct=0.1, momentum=0.95, weight_decay=0.001
    # Best Accuracy: 99.32%
    # -------------------22------------------------------33-------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.19%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=175, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.16%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=35, final_div=225, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.28%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.35, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.10%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.45, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.19%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.4, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.38%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.6, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.22%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.85, weight_decay=0.0001
    # Best Accuracy: 99.20%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.95, weight_decay=0.0001
    # Best Accuracy: 99.08%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=5e-05
    # Best Accuracy: 99.20%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0002
    # Best Accuracy: 99.09%
    # --------------------------------------------------------------------------------

    # Final Results: with 20 epochs OneCycle and image rotation 16 p=0.15
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=20, final_div=50, warmup_pct=0.3, momentum=0.9, weight_decay=1e-05
    # Best Accuracy: 99.43%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=100, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.23%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.3, initial_div=25, final_div=100, warmup_pct=0.3, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.30%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.2, initial_div=20, final_div=50, warmup_pct=0.4, momentum=0.95, weight_decay=1e-05
    # Best Accuracy: 99.38%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.1, initial_div=10, final_div=25, warmup_pct=0.2, momentum=0.85, weight_decay=5e-05
    # Best Accuracy: 99.33%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.5, initial_div=50, final_div=300, warmup_pct=0.1, momentum=0.95, weight_decay=0.001
    # Best Accuracy: 99.33%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.26%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=25, final_div=175, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.40%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=35, final_div=225, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.24%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.35, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.34%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.45, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.38%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.4, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.30%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.6, momentum=0.9, weight_decay=0.0001
    # Best Accuracy: 99.31%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.85, weight_decay=0.0001
    # Best Accuracy: 99.34%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.95, weight_decay=0.0001
    # Best Accuracy: 99.33%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=5e-05
    # Best Accuracy: 99.18%
    # --------------------------------------------------------------------------------
    # Parameters: max_lr=0.4, initial_div=30, final_div=200, warmup_pct=0.5, momentum=0.9, weight_decay=0.0002
    # Best Accuracy: 99.24%
    # --------------------------------------------------------------------------------