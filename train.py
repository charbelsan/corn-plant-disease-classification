from torchsummary import summary              # for getting the summary of our modelimport numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import argparse
import torch                    # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from utils import *
from model import *



parser = argparse.ArgumentParser(description='corn leaf disease classification model training')
parser.add_argument('--save', action='store_true', default=False,
                    help='save model to current directory')

def main():
    global args
    args = parser.parse_args()

    #config from config file
    config=read_config_file('config.json')

    #corn data extract with data_processing script  are in dst='./corn_data' directory
    dst='./corn_data'
    #train and test set directory path
    train_dir=dst+'/train'
    valid_dir=dst+'/valid'
    print('train data directory is :  ', train_dir)
    print('validation data directory is : ',  valid_dir)

    # datasets for validation and training
    train = ImageFolder(train_dir, transform=transforms.ToTensor())
    valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

    #dataset image shape
    img, label = train[0]
    print(img.shape, label)


    # total number of classes in train set
    print('total number of classes in trainset is : ', len(train.classes))



    #display the first image in the dataset
    display_img(*train[0],train)
    
    # Setting the seed value
    random_seed = config['random_seed']
    torch.manual_seed(random_seed)
    
    # setting the batch size
    batch_size = config['batch_size']
    # DataLoaders for training and validation
    train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)
    
    
    device = get_default_device()
    print('default device is : ', device)
    
    # defining the model and moving it to the GPU
    model = to_device(ResNet9(3, len(train.classes)), device) 
    print(model)
    
    # getting summary of the model
    INPUT_SHAPE = (3, 256, 256)
    print(summary(model.cuda(), (INPUT_SHAPE)))
    
    #config 
    epochs = config['epochs']
    max_lr = config['max_lr']
    grad_clip = config['grad_clip']
    weight_decay = config['weight_decay']

    # adam algorithm for optimizer
    opt_func = torch.optim.Adam
    
    history = [evaluate(model, valid_dl)]
    print(history)
    history += fit_OneCycle(epochs, max_lr, model, train_dl,
					valid_dl,grad_clip=grad_clip,
					weight_decay=1e-4,opt_func=opt_func)
    
    
    #plot accuracy as a function of epoch number
    plot_accuracies(history)
    
    #train and validation losses
    plot_losses(history)
    
    
    if args.save:
        # saving to the kaggle working directory
        PATH = './plant-disease-model.pth'  
        torch.save(model.state_dict(), PATH)




if __name__ == '__main__':
    main()
