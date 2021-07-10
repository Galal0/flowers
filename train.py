import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from workspace_utils import active_session
from PIL import Image
import numpy as np
import argparse




parser = argparse.ArgumentParser(description='Train a Neural Network')
parser.add_argument('data_dir', help ='data directory')
parser.add_argument('--save_dir', help = 'checkpoint directory', type = str, default = '/home/workspace/ImageClassifier')
parser.add_argument('--arch', help = 'model used: vgg11 or AlexNet', type = str, default = 'vgg11')
parser.add_argument('--learning_rate', help = 'model learnig rate', type = float, default = 0.001)
parser.add_argument('--hidden_units', help = 'number of hidden parameters', type = int, default = 5000)
parser.add_argument('--epochs', help = 'number of epochs', type = int, default = 7)
parser.add_argument('--gpu', help = 'using gpu', action='store_true', default = 'cpu')
args = parser.parse_args()



data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'






train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)







lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu
arch = args.arch
if device == True:
    device = 'cuda'
if arch == 'AlexNet':    
    model = models.alexnet(pretrained=True)
    output = 9216
else:
    output = 25088
    model = models.vgg11(pretrained=True)

    
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(output, hidden_units),
                                 nn.ReLU(),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
model.to(device)





with active_session():
 
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()        
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    logps = model(images)
                    valid_loss += criterion(logps, labels)
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
            model.train()
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                  "valid Accuracy: {:.3f}".format(accuracy/len(validloader)))

                    
                    
                    
                    
model.class_to_idx = train_data.class_to_idx
checkpoint = {'model' : models.vgg11(pretrained=True),
              'classifier': model.classifier,
              'model_dict': model.state_dict(),
              'loss' : criterion,
              'epochs': 10,
              'batch_size' : 64,
              'lr' : 0.001,
              'model_idx' : model.class_to_idx}
save_dir = args.save_dir
torch.save(checkpoint, save_dir + '/checkpoint.pth')
