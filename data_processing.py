# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 16:22:26 2022

@author: Charbel
"""

"""
The data set containing different leaves of healthy and unhealthy crops


This code is used to parse the dataset and extract corn leaf images from the dataset
"""
import os
import distutils.dir_util
import argparse


#dataset directory
dataset_dir="./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

parser = argparse.ArgumentParser(description='dataset processing')
parser.add_argument(
        "--data_dir",
        type=str,
        default=dataset_dir,
        help="Path to the data directory",
    )
args = parser.parse_args()
#Loading the data

#dataset directory
data_dir=args.data_dir
train_dir = data_dir + "/train"
valid_dir = data_dir + "/valid"
diseases = os.listdir(train_dir)

# printing the disease names
print('disease names:  ',  diseases)


plants = []
for plant in diseases:
    if plant.split('___')[0] not in plants:
        plants.append(plant.split('___')[0])
print('different plants names /n', plants  )

#only maize diseases interest us, named 'Corn_(maize)' in the dataset
corn_diseases=[]
for plant_d in diseases:
    if plant_d.split('___')[0]=='Corn_(maize)':
        corn_diseases.append(plant_d)

#number of corn diseases
print('Corn deaseases find: ', corn_diseases)


#number of corn diseases
print('number of corn diseases  :  ', len(corn_diseases)-1)

#number of images for each category
n=0
for disease in diseases:
     if disease.split('___')[0]=='Corn_(maize)':
        print(disease+' : '+str(len(os.listdir(train_dir + '/' + disease))))
        n+=len(os.listdir(train_dir + '/' + disease))
print(n,'  images in total for training')

#create a folder for corn data
dst='./corn_data'
#os.mkdir(dst)
for corn_disease in corn_diseases:
    src_train_dir=train_dir+'/'+corn_disease
    src_valid_dir=valid_dir+'/'+corn_disease
    print(dst+'/train/'+corn_disease)
    distutils.dir_util.copy_tree(src_train_dir, dst+'/train/'+corn_disease)
    distutils.dir_util.copy_tree(src_valid_dir, dst+'/valid/'+corn_disease)
