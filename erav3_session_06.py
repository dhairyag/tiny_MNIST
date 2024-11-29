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
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),  # 4*3*3*8 + 8 = 296 params
            nn.ReLU(),
            nn.BatchNorm2d(8),              # 16 params
            nn.Conv2d(8, 8, 3, padding=1),  # 8*3*3*8 + 8 = 584 params
            nn.ReLU(),
            nn.BatchNorm2d(8),              # 16 params
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1), # 8*3*3*12 + 12 = 876 params
            nn.ReLU(),
            nn.BatchNorm2d(12),             # 24 params
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.05)
        )
        
        # 12 channels * 3 * 3 = 108 neurons after three max pools (28->14->7->3)
        self.fc1 = nn.Linear(12 * 3 * 3, 10)  # 108*10 + 10 = 1090 params
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 12 * 3 * 3)  # Flatten
        x = F.dropout(x, p=0.0)
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
    # A.ShiftScaleRotate(
    #     #shift_limit=0.02,
    #     #scale_limit=0.05,
    #     rotate_limit=7,
    #     p=0.2,
    #     border_mode=cv2.BORDER_CONSTANT,
    #     value=0
    # ),
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
    # -----------------------------
    # Tunable parameters
    max_lr = 0.3          # Peak learning rate
    initial_div = 25      # Divisor for initial learning rate (max_lr / initial_div)
    final_div = 100       # Divisor for final learning rate
    warmup_pct = 0.3      # Percentage of training for warmup
    momentum = 0.9        # Momentum for SGD
    weight_decay = 1e-4   # L2 regularization
    num_epochs = 15       # Total number of epochs

    #max_lr,	initial_div,	final_div,	warmup_pct,	momentum,	weight_decay
    # 0.3,	25,	100,	0.3,	0.9,	1e-4
    # 0.2,	20,	50,	0.4,	0.95,	1e-5
    # 0.1,	10,	25,	0.2,	0.85,	5e-5
    # 0.4,	30,	200,	0.5,	0.9,	1e-4
    # 0.5,	50,	300,	0.1,	0.95,	1e-3
    # Define parameter combinations
    param_combinations = [
        # max_lr, initial_div, final_div, warmup_pct, momentum, weight_decay
        (0.3, 25, 100, 0.3, 0.9, 1e-4),
        (0.2, 20, 50, 0.4, 0.95, 1e-5),
        (0.1, 10, 25, 0.2, 0.85, 5e-5),
        (0.4, 30, 200, 0.5, 0.9, 1e-4),
        (0.5, 50, 300, 0.1, 0.95, 1e-3),
        # Additional combinations
        (0.3, 20, 50, 0.3, 0.9, 1e-5),
        (0.3, 10, 25, 0.3, 0.9, 5e-5),
        (0.3, 30, 200, 0.3, 0.9, 1e-4),
        (0.2, 25, 100, 0.4, 0.95, 1e-4),
        (0.2, 10, 25, 0.4, 0.95, 5e-5),
        (0.2, 30, 200, 0.4, 0.95, 1e-4),
        (0.1, 25, 100, 0.2, 0.85, 1e-4),
        (0.1, 20, 50, 0.2, 0.85, 1e-5),
        (0.1, 30, 200, 0.2, 0.85, 1e-4),
        (0.4, 25, 100, 0.5, 0.9, 1e-4),
        (0.4, 20, 50, 0.5, 0.9, 1e-5),
        (0.4, 10, 25, 0.5, 0.9, 5e-5),
        (0.5, 25, 100, 0.1, 0.95, 1e-4),
        (0.5, 20, 50, 0.1, 0.95, 1e-5),
        (0.5, 10, 25, 0.1, 0.95, 5e-5),
    ]

    # Store results
    results = []
    num_epochs = 15

    # Run training for each parameter combination
    for params in param_combinations:
        max_lr, initial_div, final_div, warmup_pct, momentum, weight_decay = params
        print(f"\nTraining with parameters:")
        print(f"max_lr: {max_lr}, initial_div: {initial_div}, final_div: {final_div}")
        print(f"warmup_pct: {warmup_pct}, momentum: {momentum}, weight_decay: {weight_decay}")

        # Initialize model
        model = Net().to(device)
        
        # Optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=max_lr / initial_div,
            momentum=momentum,
            weight_decay=weight_decay
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=warmup_pct,
            div_factor=initial_div,
            final_div_factor=final_div
        )

        # Track best accuracy for this combination
        best_accuracy = 0.0

        # Training Loop
        for epoch in range(1, num_epochs + 1):
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}: Current learning rate: {current_lr:.6f}')
            
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
    plt.title('Best Test Accuracy for Different Parameter Combinations')
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
