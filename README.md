# corn-plant-disease-classification
corn plant disease classification with pytorch


# Requierement
* Pytorch
* torchvision
* torchsummary :     to print the model's summary in keras style
* ray-tune   :  for hyperparameters tuning
#### Dataset
 base dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. 
 We only need corn pictures,the `data_proceesing.py` script creates a new dataset containing only corn images, keeping the same structure as the initial dataset
 
 
```bash
data_processing.py [-h] [--data_dir DATA_DIR]

dataset processing

options:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Path to the data directory
 
```
data_dir is the path to the dataset folder which contains Train and valid folders
if the path to the dataset is not specified, data_dir=dataset_dir="./data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

a `corn_data` folder is created in the current project folder.

# Taining
the architecture of the model to train is defined in the model.py file and configuration parameters in config.json

Train the model
 ```bash
    python train.py
```
Train and save in current project folder.
```bash
    python train.py --save
```

batch size and epochs are configurable with `config.json`
Instead of using a fixed learning rate, we will use a learning rate scheduler, max_lr in config.json is max learning rate. the other parameters are Weight decay ,Gradient clipping,

this strategy is better explained on my kaggle notebook about this project.

 [find the notebbok here](https://www.kaggle.com/code/charbelsan/classification-des-maladies-du-mais)

