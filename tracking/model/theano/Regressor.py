# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:27:13 2016

@author: MindLab
"""

from tracking.model.core.Regressor import Regressor

class Regressor(Regressor):  
    
    # layers = [(tracking.model.core.CnnLayer, boolean)]
    def __init__(self, layers):
        self.layers = layers
        
        
    # frame.shape = (batchSize, channels, height, width)
    def forward(self, prevPos, data):
        fmap = data
        
        for layer in self.layers:
            fmap = layer.forward(fmap)
        
        return fmap
        
    
    def getParams(self):
        params = []
        
        for layer in self.layers:
            if layer.trainable:
                params.extend(layer.getParams())

        return params
        
        
    def getOutputDim(self, inDims):
        outDims = inDims
        
        for layer in self.layers:
            outDims = layer.getOutputDim(outDims)
        
        return outDims
    
    