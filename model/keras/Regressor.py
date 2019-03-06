# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 22:38:33 2016

@author: MindLab
"""

from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from tracking.model.keras.Module import Module

class Regressor(Module):
    
    def __init__(self, layers):
        self.build(layers)
        
    
    def build(self, layers):
        model = Sequential()
        
        for layer in layers:
            model.add(layer)
        
        self.model = TimeDistributed(model)
        
    
    def setTrainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable
            
    
    def getOutputShape(self):
        outputShape = self.model.layer.output_shape
        
        return outputShape