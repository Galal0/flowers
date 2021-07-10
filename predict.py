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


parser = argparse.ArgumentParser(description='Predict using neural network')
parser.add_argument('path', help ='path to image')
parser.add_argument('checkpoint', help ='path to trained model')
parser.add_argument('--topk', help ='number of most probable classes', type = int, default = 5)
parser.add_argument('--category_names', help ='classes to category file', default = 'cat_to_name.json')
parser.add_argument('--gpu', help = 'using gpu', action='store_true', default = 'cpu')
args = parser.parse_args()


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
def load_model(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_dict'])
    criterion = checkpoint['loss']
    epochs = checkpoint['epochs']
    idx = checkpoint['model_idx']
    return [model, criterion, idx, epochs]





def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as im:
        w, h = im.size
        ratio = w / h
        if w > h:
            im = im.resize((int(256*ratio), 256))
        else:
            im = im.resize((256, int(256/ratio)))
        left = int((w - 224) / 2)
        right = left + 224
        upper = int((h - 224) / 2)
        lower = upper + 224
        im = im.crop((left, upper, right,lower))
        im = np.array(im)/255
        mean = [0.485, 0.456, 0.406]
        sd = [0.229, 0.224, 0.225]
        im = (im - mean) / sd
        im = im.transpose(2,0,1)
    return im





def predict(image_path, path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = args.gpu
    if device == True:
        device = 'cuda'
    model = load_model(path)[0]
    model.eval()
    model.to(device)
    image = process_image(image_path)
    image = torch.from_numpy(image)
    image.unsqueeze_(0)
    image = image.to(device, dtype=torch.float)
    logps = model(image)
    ps = torch.exp(logps)
    top_p, top_idx = ps.topk(topk, dim=1)
    idx = load_model(path)[2]
    inverted_idx = {value : key for (key, value) in idx.items()}
    classes = []
    for i in top_idx:
        for j in i:
            classes.append(inverted_idx[int(j)])
    return top_p[0], classes






def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    




def view (image_path, ps, classes):

    with torch.no_grad():
        fig = plt.figure(figsize=(6,6))
        ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
        ax2 = plt.subplot2grid((15,9), (9,2), colspan=5, rowspan=5)
        image = Image.open(image_path)
        ax1.axis('off')
        name = classes[np.argmax(ps)]
        ax1.set_title(cat_to_name[name])
        ax1.imshow(image)
        names = []
        for i in classes:
            names.append(cat_to_name[i])
        ax2.set_yticks(np.arange(5))
        ax2.set_yticklabels(names)
        ax2.set_xlabel('Probability')
        ax2.barh(np.arange(5), ps)
        plt.show()
        
path = args.path
checkpoint = args.checkpoint
topk = args.topk


ps, classes = predict(path, checkpoint, topk)

flowers = []
for c in classes:
    flowers.append(cat_to_name[c])
print ('Probabilities: ', ps)
print('Classes: ', classes)
print('Flowers: ', flowers)

