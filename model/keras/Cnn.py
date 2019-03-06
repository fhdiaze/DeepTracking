# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:11:10 2016

@author: MindLab
"""

from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from tracking.model.keras.Module import Module

class Cnn(Module):
    
    def __init__(self, input, layers):
        self.build(input, layers)
        
    
    def build(self, input, layers):
        output = input
        
        for layer in layers:
            output = layer(output)
        
        model = Model(input=input, output=output)
        self.model = TimeDistributed(model)
    
    
    def getOutputShape(self):
        outputShape = self.model.layer.output_shape
        
        return outputShape
        
        
    def setTrainable(self, trainable):
        for layer in self.model.layers:
            layer.trainable = trainable