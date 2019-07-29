# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:13:53 2016

@author: fhdiaze
"""

from tracking.model.core.Cnn import Cnn

class Cnn(Cnn):
    
    # layers = [(tracking.model.core.CnnLayer, boolean)]
    def __init__(self, layers):
        self.layers = layers
        
        
    # layer = tracking.model.core.CnnLayer
    def addLayer(self, layer):
        self.layers.append(layer)
        
        
    # frame.shape = (batchSize, channels, height, width)
    def forward(self, frame):
        fmap = frame
        
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