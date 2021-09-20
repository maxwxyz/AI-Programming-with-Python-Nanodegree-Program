import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import PIL
from PIL import Image

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='directory for data:', default = './flowers')
parser.add_argument('--arch', type = str, default = 'vgg16', choices=['vgg16', 'densenet121'], help = 'Only vgg16 and densenet121 supported.')
parser.add_argument('--save_dir', dest= 'save_dir', type = str, default = './checkpoint.pth', help = 'Folder where the model is saved: default is current.')
parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Gradient descent learning rate')
parser.add_argument('--hidden_layer', type = int, action= 'store', dest = 'hidden_layer', default = 25088, help = 'Number of hidden units #1 for classifier.')
parser.add_argument('--hidden_layer2', type = int, action= 'store', dest = 'hidden_layer2', default = 4096, help = 'Number of hidden units #2 for classifier.')
parser.add_argument('--output_layer', type = int, action= 'store', dest = 'output_layer', default = 102, help = 'Number of output units for classifier.')
parser.add_argument('--epochs', type = int, help = 'Number of epochs', default = 3)
parser.add_argument('--device', action='store_true', help='Use this flag if you want to use GPU for prediction')

parser.add_argument('--image_path', type=str, help='path of image to be predicted', default = 'flowers/test/1/image_06743.jpg')
parser.add_argument('--topk', type=int, default=5, help='display top k probabilities')
parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json', help='provide path to category mapping file')
parser.add_argument('--checkpoint_path', type=str, help='checkpoint file to be used', default = 'checkpoint.pth')
args = parser.parse_args()

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
           
    return model, checkpoint['class_to_idx']

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
      
    image = Image.open(image)
        
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    image = preprocess(image)
    
    return image

def predict(image_path, model, topk, class_idx_dict):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    img = process_image(image_path)
    model.to(device)
    img = img.to(device)
    
    img_classes_dict = {v: k for k, v in class_to_idx.items()}
    
    model.eval()
    
    with torch.no_grad():
        img.unsqueeze_(0)
        output = model.forward(img)
        ps = torch.exp(output)
        probs, classes = ps.topk(topk)
        probs, classes = probs[0].tolist(), classes[0].tolist()
        
        return_classes = []
        for c in classes:
            return_classes.append(img_classes_dict[c])
            
        return probs, return_classes