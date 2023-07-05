import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from hyperparameters import *

def download_and_load_mnist(batch_size, shuffle=False):
    # Define the paths for the dataset files
    data_path = os.path.expanduser("~/.torch/datasets/")
    mnist_path = os.path.join(data_path, "MNIST")

    # Check if the files exist, and if not, set download flag to True
    if not os.path.exists(mnist_path):
        print("MNIST dataset not found. Downloading...")
        download_flag = True
    else:
        print("MNIST dataset already present. Loading...")
        download_flag = False

    # rescale in [-1, 1] and convert to tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 1)])

    # Load the dataset and store it in the folder
    train_set = datasets.MNIST(mnist_path, train=True, download=download_flag, transform=transform)
    test_set = datasets.MNIST(mnist_path, train=False, download=download_flag, transform=transform)
    

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    

    

    
    print("MNIST dataset loaded successfully as torch tensors with corresponding labels!")

    return train_loader, test_loader

def train(encoder, decoder, train_loader, criterion, optimizer, device):
    encoder.train()
    decoder.train()
    running_loss = 0.0

    for data, _ in train_loader:
        print(data.size())

        data = data.to(device)
        optimizer.zero_grad()
        latent = encoder(data)
        #### #### #### #### #### ####
        #latent = torch.abs(latent)
        min_val, _= torch.min(latent,dim=1)
        latent=latent-min_val.unsqueeze(1)
        latent = F.normalize(latent, dim=1)
        #### #### #### #### #### ####
        outputs = decoder(latent)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)

def eval(encoder, decoder, test_loader, criterion, device):
    encoder.eval()
    decoder.eval()
    running_loss = 0.0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            latent = encoder(data)
            #### #### #### #### #### ####
            #latent = torch.abs(latent)
            min_val, _= torch.min(latent,dim=1)
            latent=latent-min_val.unsqueeze(1)
            latent = F.normalize(latent, dim=1)
            #### #### #### #### #### ####
            outputs = decoder(latent)
            loss = criterion(outputs, data)
            running_loss += loss.item()

    return running_loss / len(test_loader)

def which_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: {}".format(device))

    return device

def sample_latent_space(encoder, decoder, device, n_samples=10):
    encoder.eval()
    decoder.eval()

    # Sample random points in the latent space
    latent_space_samples = torch.randn(n_samples, LATENT_DIM).to(device)

    #### #### #### #### #### ####
    #latent_space_samples = torch.abs(latent_space_samples)
    min_val, _= torch.min(latent_space_samples,dim=1)
    latent_space_samples=latent_space_samples-min_val.unsqueeze(1)
    latent_space_samples = F.normalize(latent_space_samples, dim=1)
    #### #### #### #### #### ####

    # Decode the samples from the latent space
    decoded_samples = decoder(latent_space_samples).cpu().detach().numpy()

    return decoded_samples

def plot_decoded_samples(decoded_samples):
    # Reshape the decoded samples to their original dimensions (e.g., 28x28 for MNIST)
    decoded_images = decoded_samples.reshape(-1, 28, 28)

    # Plot the images
    _, axes = plt.subplots(1, len(decoded_images), figsize=(12, 8))

    if len(decoded_images) == 1:
        axes.imshow(decoded_images[0], cmap='gray')
        axes.axis('off')
    else:
        for i, img in enumerate(decoded_images):
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')

    plt.show()

def download_and_load_mnist_sample(labels):
    # Define the paths for the dataset files
    data_path = os.path.expanduser("~/.torch/datasets/")
    mnist_path = os.path.join(data_path, "MNIST")

    # Check if the files exist, and if not, set download flag to True
    if not os.path.exists(mnist_path):
        print("MNIST dataset not found. Downloading...")
        download_flag = True
    else:
        print("MNIST dataset already present. Loading...")
        download_flag = False

    # Define the transformations to be applied to the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 1)])

    # Load the dataset and filter by labels
    train_set = datasets.MNIST(mnist_path, train=True, download=download_flag, transform=transform)
    train_set = list(filter(lambda x: x[1] in labels, train_set))
    test_set = datasets.MNIST(mnist_path, train=False, download=download_flag, transform=transform)
    test_set = list(filter(lambda x: x[1] in labels, test_set))

    # Convert the list of images and labels to a torch tensor
    train_images, train_labels = zip(*train_set)
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    
    test_images, test_labels = zip(*test_set)
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)

    print(f"MNIST dataset with labels {labels} loaded successfully as torch tensors!")

    return train_images, train_labels, test_images, test_labels