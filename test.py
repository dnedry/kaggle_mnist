#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

import data_loader as dl
import model

DATA_OPTIONS = {
    'DIRECTORY': r'X:/kaggle_mnist_pytorch/kaggle_mnist',
    'SPLIT_SIZE': 0.15, # Ratio of validation/train data
    'SEED': 42
}

def main():
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data object
    data = dl.Dataset(DATA_OPTIONS, device)

    # Pytorch model
    mlp = model.MLP()

    print('hi')

if __name__ == "__main__":
    main()