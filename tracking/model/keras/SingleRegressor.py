# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 17:34:27 2017

@author: MindLab
"""

from keras.models import Model
from tracking.model.keras.Module import Module

class SingleRegressor(Module):
    
    def __init__(self, input, layers):
        self.build(input, layers)
        
    
    def build(self, input, layers):
        output = input
        
        for layer in layers:
            output = layer(output)
        
        self.model = Model(input=input, output=output)
        
    
    def setTrainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable