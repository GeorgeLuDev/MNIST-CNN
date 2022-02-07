import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ConvNet import ConvNet 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def inference(model, device, test_loader):
    # Set model to evaluation mode
    model.eval()

    # test loop
    with torch.no_grad():
        # extract batch and send to device
        batch_sample = next(iter(test_loader))
        data, target = batch_sample
        data, target = data.to(device), target.to(device)

        # forward pass
        output = model(data)

        # calculate number correct
        pred = output.argmax(dim=1, keepdim=True)
        pred = pred.reshape(-1)

        return data.cpu().detach().numpy(), pred.cpu().detach().numpy(), target.cpu().detach().numpy()
    


def run_main():
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # Create Transform based of datasets mean and std
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    dataset2 = datasets.MNIST('./data/', train=False,transform=transform)

    batch_size = 10

    # Convert datasets to dataloaders
    test_loader = DataLoader(dataset2, batch_size = batch_size, shuffle=False, num_workers=2)
    
    # Initialize the model and send to device 
    model = ConvNet().to(device)

    # Initalize loss function
    criterion = nn.CrossEntropyLoss()

    # model paths
    model_path = "./models/"
    best_model_path = model_path + "best_model.pth"

    # load checkpoint
    if os.path.exists(best_model_path): 
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    images, predictions, targets = inference(model, device, test_loader)
    images = np.squeeze(images)

    row, col = 5,2

    plt.figure(figsize=(row*2.5, col*2.25))
    plt.gray()

    for index, (image,prediction,target) in enumerate(zip(images, predictions, targets)):
        plt.subplot(col,row,index+1)
        plt.imshow(image,cmap="gray")
        plt.title(f'Target {target} | Prediction {prediction}')
        plt.axis('off')
        
    plt.savefig("graph/results.jpg")
    plt.show()

if __name__ == '__main__':
    run_main()
    
    