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

def train(model, device, train_loader, optimizer, criterion):
    # Set model to training mode
    model.train()

    # keep track of losses and correct
    losses = []
    correct = 0

    # training loop
    for batch_idx, batch_sample in enumerate(train_loader):
        # extract batch and send to device
        data, target = batch_sample 
        data, target = data.to(device), target.to(device)

        # zero out the grad
        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # calculate loss
        loss = criterion(output, target)

        # calculate gradient
        loss.backward()

        # back prop
        optimizer.step()

        # add loss to list
        losses.append(loss.item())

        # calculate number correct
        pred = output.argmax(dim=1, keepdim=True)
        pred = pred.reshape(-1)
        correct += torch.sum(pred == target)

    # calculate loss and accuracy
    train_loss = float(np.mean(losses))
    train_acc = float(100 * correct / len(train_loader.dataset))
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(float(np.mean(losses)), correct, len(train_loader.dataset), train_acc))
    return round(train_loss,2), round(train_acc,2)

def test(model, device, test_loader, criterion):
    # Set model to evaluation mode
    model.eval()
    
    # keep track of losses and correct
    losses = []
    correct = 0

    # test loop
    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(test_loader):
            # extract batch and send to device
            data, target = batch_sample
            data, target = data.to(device), target.to(device)

            # forward pass
            output = model(data)

            # calculate loss
            loss = criterion(output,target)

            # add loss to list
            losses.append(loss.item())

            # calculate number correct
            pred = output.argmax(dim=1, keepdim=True)
            pred = pred.reshape(-1)
            correct += torch.sum(pred == target)

    # calculate loss and accuracy
    test_loss = float(np.mean(losses))
    test_accuracy = float(100. * correct / len(test_loader.dataset))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), test_accuracy))
    return round(test_loss,2), round(test_accuracy,2)

def save_model(epoch, model, optimizer, histry_df, best_accuracy, path):
    torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'history_df' : histry_df,
    'best_acc' : best_accuracy
    }, path)


def run_main():
    # Parameters
    learning_rate = 3e-4
    log_dir = "results"
    batch_size = 100
    num_epochs = 125

    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    # Create dataframe to track accuracies and losses
    df = pd.DataFrame(columns = ["train_loss","train_accuracy","test_loss","test_accuracy"])

    # Create Transform based of datasets mean and std
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    # Load datasets for training and testing
    dataset1 = datasets.MNIST('./data/', train=True, download=True,transform=transform)
    dataset2 = datasets.MNIST('./data/', train=False,transform=transform)

    # Convert datasets to dataloaders
    train_loader = DataLoader(dataset1, batch_size = batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset2, batch_size = batch_size, shuffle=False, num_workers=2)
    
    # Initialize the model and send to device 
    model = ConvNet().to(device)

    # Initalize loss function
    criterion = nn.CrossEntropyLoss()

    # Initalize optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0
    starting_epoch = 0
    model_path = "./models/"
    best_model_path = model_path + "best_model.pth"
    latest_model_path = model_path + "latest_model.pth"

    # load checkpoint
    if os.path.exists(latest_model_path): 
        checkpoint = torch.load(latest_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        df = checkpoint['history_df']
        best_accuracy = checkpoint['best_acc']

    # for each epoch run train and test
    for epoch in range(starting_epoch, num_epochs):
        print(f'current epoch {epoch + 1}')
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        
        df = df.append({"train_loss" : train_loss, "train_accuracy" : train_accuracy, "test_loss" : test_loss, "test_accuracy" : test_accuracy}, ignore_index = True)

        if (test_accuracy > best_accuracy):
            best_accuracy = test_accuracy
            save_model(epoch, model, optimizer, df, best_accuracy, best_model_path)

        save_model(epoch, model, optimizer, df, best_accuracy, latest_model_path)

    
    print("Training and evaluation finished")
    print("Best Accuracy is {:2.2f}".format(best_accuracy))
    
    # print history
    print(df)

    # write to file
    path = "./" + log_dir + "/model_results.csv"
    df.to_csv(path,index=False)
    
    
if __name__ == '__main__':
    run_main()
    
    