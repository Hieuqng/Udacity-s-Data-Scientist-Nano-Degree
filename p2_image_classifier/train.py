import numpy as np

# Pytorch
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms, utils

# Miscellaneous
import os
import time
import json
from PIL import Image
import argparse

def load_model(hidden_layer_size, arch):
    model = getattr(models, arch)(pretrained=True)
    
    # Get the size of last hidden layer to rebuild new classifier
    if 'densenet' in arch:
        num_features = model.classifier.in_features
    elif 'resnet' in arch:
        num_features = model.fc.in_features
    elif 'vgg' in arch:
        # We want to add new layers to classifier, so get the last output size
        num_features = model.classifier[-1].out_features
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Define new classifier to classify 102 classes of flowers
    classifier = nn.Sequential(
        nn.Linear(in_features=num_features, out_features=hidden_layer_size),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(in_features=hidden_layer_size, out_features= 102),
        nn.LogSoftmax(dim=1)
    )
    
    if 'resnet' in arch:
        model.fc = classifier
    elif 'vgg' in arch:
        # Unfreeze classifier because we just add new layers to it
        for param in model.classifier.parameters():
            param.requires_grad = True
        model.classifier = nn.Sequential(model.classifier, classifier)
    else:
        model.classifier = classifier
        
    return model


def accuracy_score(model, data, criterion, device):
    loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X, y in data:
            X, y = X.to(device), y.to(device)
            output = model.forward(X)
            loss += criterion(output, y).item()
            total += y.shape[0]
            correct += (y.data == output.max(dim=1)[1]).sum().item()
            
    return loss / len(data), correct / total 


def train(model, dataloaders, criterion, optimizer, epochs, device, verbose=True):
    for e in range(epochs):
        model.train()
        epoch_loss = 0
        
        for X_train, y_train in dataloaders['train']:
            X_train, y_train = X_train.to(device), y_train.to(device)

            model.zero_grad()
            output = model.forward(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        val_loss, val_acc = accuracy_score(model, dataloaders['valid'], criterion, device)
        
        if verbose:
            print("Epoch: {}/{}..  ".format(e + 1, epochs),
                  "Epoch loss: {:.4f}..  ".format(epoch_loss / len(dataloaders['train'])),
                  "Validation loss: {:.4f}..  ".format(val_loss),
                  "Validation Accuracy: {:.4f}..  ".format(val_acc))   

            
def main():
    parser = argparse.ArgumentParser(description='Training flower images classifier')
    parser.add_argument('data_dir', action='store', help='Directory of datasets')
    parser.add_argument('--gpu', action='store', dest='gpu', help='Use gpu or not', default=True)
    parser.add_argument('--model', action='store', dest='model_name', help='Name of the architecture', default='densenet121')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', help='Store learning rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', help='Choose hidden units of classifier', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', help='Number of epochs to train', type=int, default=5)
    parser.add_argument('--batch_size', action='store', dest='batch_size', help='Number of batches', type=int, default=16)
    parser.add_argument('--save_dir', action='store', dest='save_dir', help='Directory to save checkpoint', default='ImageClf.pth')
    
    args = parser.parse_args()
    gpu = args.gpu
    if gpu and torch.cuda.is_available():
        use_gpu = True
        device = torch.device("cuda:0")
    else:
        use_gpu = False
        device = torch.device("cpu")
    
    #Get inputs from args
    data_dir = args.data_dir
    model_name = args.model_name
    batch_size = args.batch_size
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    epochs = args.epochs
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    data_transforms = {'train': train_transforms,
                      'valid' : test_transforms,
                      'test' : test_transforms}

    images_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transforms[x])
                       for x in ['train', 'valid', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(images_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'valid', 'test']}

    dataset_sizes = {x: len(images_datasets[x]) for x in ['train', 'valid', 'test']}



    # Build model
    model = load_model(hidden_units, model_name)
    model.to(device)

    # Choose loss function and optimizer
    criterion = nn.NLLLoss()
    if 'resnet' in model_name:
        optimizer = optim.Adam(params=model.fc.parameters(), lr=learning_rate)
    else:
        optimizer = optim.Adam(params=model.classifier.parameters(), lr=learning_rate)
    # Train model
    train(model, dataloaders, criterion, optimizer, epochs, device)

    # Save model
    save_dir = args.save_dir
    model.class_to_idx = images_datasets['train'].class_to_idx
    state = {
        "epochs" : epochs,
        "batch_size" : batch_size,
        "learning_rate" : learning_rate,
        "clf_output_size" : hidden_units,
        "opt_state_dict" : optimizer.state_dict(),
        "criterion" : criterion,
        "class_to_idx" : model.class_to_idx,
        "state_dict": model.state_dict()
    }
    torch.save(state, save_dir)
    
    
if __name__ == '__main__':
    main()