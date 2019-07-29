# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:34:57 2016

@author: MindLab
"""
from tracking.model.theano.Layer import Layer

class NonLinearityLayer(Layer):
    
    def __init__(self, name, nonlinearity):
        Layer.__init__(self, name, False)
        self.nonlinearity = nonlinearity
        
    def forward(self, inData):
        fmap = self.nonlinearity(inData)
        
        return fmap
        
        
    def getParams(self):
        
        return []
        
        
    def getOutputDim(self, inDims):
        
        return inDims