# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:14:44 2016

@author: MindLab
"""

import h5py
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from tracking.model.keras.Cnn import Cnn
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D

class Vgg(Cnn):
    
    def __init__(self, modelPath, layerKey):
        self.layerKey = layerKey
        self.buildModel(modelPath)        
    
        
    def buildModel(self, modelPath):
        layers = []
        
        layers.append(ZeroPadding2D((1,1), input_shape=(3, 224, 224), name="input"))
        layers.append(Conv2D(64, kernel_size=(3 , 3), activation='relu', name='conv1_1'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(64, kernel_size=(3 , 3), activation='relu', name='conv1_2'))
        layers.append(MaxPooling2D((2, 2), strides=(2, 2)))
        
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(128, kernel_size=(3 , 3), activation='relu', name='conv2_1'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(128, kernel_size=(3 , 3), activation='relu', name='conv2_2'))
        layers.append(MaxPooling2D((2, 2), strides=(2, 2)))
        
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(256, kernel_size=(3 , 3), activation='relu', name='conv3_1'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(256, kernel_size=(3 , 3), activation='relu', name='conv3_2'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(256, kernel_size=(3 , 3), activation='relu', name='conv3_3'))
        layers.append(MaxPooling2D((2, 2), strides=(2, 2)))
        
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(512, kernel_size=(3 , 3), activation='relu', name='conv4_1'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(512, kernel_size=(3 , 3), activation='relu', name='conv4_2'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(512, kernel_size=(3 , 3), activation='relu', name='conv4_3'))
        layers.append(MaxPooling2D((2, 2), strides=(2, 2)))
        
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(512, kernel_size=(3 , 3), activation='relu', name='conv5_1'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(512, kernel_size=(3 , 3), activation='relu', name='conv5_2'))
        layers.append(ZeroPadding2D((1, 1)))
        layers.append(Conv2D(512, kernel_size=(3 , 3), activation='relu', name='conv5_3'))
        layers.append(MaxPooling2D((2, 2), strides=(2, 2)))
                
        layers.append(Flatten(name='flat'))
        layers.append(Dense(4096, activation='relu', name='fc6'))
        layers.append(Dropout(0.5, name='dopout0'))
        layers.append(Dense(4096, activation='relu', name='fc7'))
        layers.append(Dropout(0.5, name='dropout1'))
        layers.append(Dense(1000, activation='softmax', name='softmax'))
        
        model = Sequential()
        
        for layer in layers:
            model.add(layer)
        
        while model.layers[-1].name != self.layerKey:
            model.pop()
            
        
        model.load_weights(modelPath, by_name=True)
            
        self.model = TimeDistributed(model)
        
        
    def getOutputShape(self):
        outDims = self.model.layer.output_shape
        
        return outDims
        
        
    def getModel(self):
        return self.model
        
    
    def setTrainable(self, trainable):
        for layer in self.model.layer.layers:
            layer.trainable = trainable