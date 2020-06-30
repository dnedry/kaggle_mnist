#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import standard packages
import os

# Import thrid party packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch


class Dataset(object):
    def __init__(self,
                 options,
                 device):

        # options
        self._options = options

        # device
        self._device = device
 
        # Datasets
        self._X_train = None
        self._y_train = None
        self._X_validation = None
        self._y_validation = None
        self._X_test = None

        self.load_data()

    @property
    def X_train(self):
        return self._X_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def X_validation(self):
        return self._X_validation

    @property
    def y_validation(self):
        return self._y_validation
      
    @property
    def X_test(self):
        return self._X_test

    def load_data(self):

        # load train data
        train_data_raw = pd.read_csv(os.path.join(self._options['DIRECTORY'], 'data', 'train.csv'))
        train_data_x = train_data_raw.drop(columns=['label'], axis=1)
        train_data_y = train_data_raw['label'].to_numpy()

        # shuffle and split
        X_train, X_validation, y_train, y_validation = train_test_split(train_data_x, train_data_y, test_size=self._options['SPLIT_SIZE'], shuffle=True, random_state=self._options['SEED'])
        
        # train data
        self._X_train = self.preprocess(X_train)
        self._X_train = torch.from_numpy(X_train.values).to(self._device)
        self._y_train = torch.from_numpy(y_train).to(self._device)

        self._X_validation = self.preprocess(X_validation)
        self._X_validation = torch.from_numpy(X_validation.values).to(self._device)
        self._y_validation = torch.from_numpy(y_validation).to(self._device)
        
        

        #print out training data
        print(f"\nTrain data x shape: {X_train.shape}")
        print(f"\nTrain data y shape: {y_train.shape}")

        # load test data
        X_test = pd.read_csv(os.path.join(self._options['DIRECTORY'], 'data', 'test.csv'))
        X_test = self.preprocess(X_test)
        self._X_test = torch.from_numpy(X_test).to(self._device)


        print('hi')

    def preprocess(self, data):
        """
        A function to preprocess feature data for use in a neural network
        """

        scaler = MinMaxScaler()
        return scaler.fit_transform(data)


