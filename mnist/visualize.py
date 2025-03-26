import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Set up data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('../data', train=False, transform=transform)

def visualize_samples(dataset, title, num_samples=5):
    # Get random samples
    indices = torch.randperm(len(dataset))[:num_samples]
    
    # Create a figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle(title)
    
    for idx, i in enumerate(indices):
        # Get image and label
        image, label = dataset[i]
        
        # Denormalize the image
        image = image * 0.3081 + 0.1307
        
        # Plot
        axes[idx].imshow(image.squeeze(), cmap='gray')
        axes[idx].set_title(f'Label: {label}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize training samples
visualize_samples(train_dataset, 'Training Set Samples')

# Visualize test samples
visualize_samples(test_dataset, 'Test Set Samples')

# Print dataset information
print("\nDataset Information:")
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print(f"\nSample image shape: {train_dataset[0][0].shape}")
print(f"Sample label type: {type(train_dataset[0][1])}") 