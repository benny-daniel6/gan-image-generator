# data_pipeline.py

import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataloader(batch_size=128, image_size=64):
    """
    Prepares the CIFAR-10 dataset and returns a DataLoader.

    Args:
        batch_size (int): The number of samples per batch.
        image_size (int): The spatial size of training images. All images will be resized to this size.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the CIFAR-10 training set.
    """
    # Define transformations for the images.
    # We resize to the desired image size, convert to a PyTorch tensor,
    # and normalize the pixel values to the range [-1, 1]. This range is
    # ideal for the Tanh activation function in the generator's output layer.
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Download the training dataset if it's not already in the './data' directory.
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)

    # Create the DataLoader.
    # The DataLoader is an iterator that provides batches of data.
    # `shuffle=True` is important for training to ensure the model sees
    # data in a random order each epoch.
    # `num_workers` can be increased to speed up data loading if you have multiple CPU cores.
    dataloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=2)

    return dataloader