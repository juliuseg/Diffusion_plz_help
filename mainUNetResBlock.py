import wandb
import torch
import os
from utils.dataset_loader import load_dataset
from configs.config import config_MNIST, config_CIFAR10, config_CELEBA
from UNetResBlock.model import UNet
from UNetResBlock.trainer import train_model
import random

config = config_CIFAR10

configDSName = config["dataset_name"]
name = f"{configDSName}_{random.randint(100000, 1000000)}"
print (f"starting run: {name}")

# Initialize W&B
logwandb = True

# Use only small subset of data (for debugging)
debugDataSize = False
modelNameTest = "_smallDS" if debugDataSize else ""

save_model = False
model_name = "ResNet"+name+modelNameTest # File that the model is saved as. Only relevant if save_model = True


if logwandb: 
    wandb.init(project=config["project_name"], name=model_name, config=config)

# Load dataset
dataset = load_dataset(config,small_sample=debugDataSize)

channels = config["image_shape"][0]
dim = config["image_shape"][1]

# Initialize model
model = UNet(in_channels=channels, dim = config["image_shape"][1], out_channels=channels)

# Train model
train_model(model, dataset, config, model_name=model_name, log=logwandb, save_model = save_model)

