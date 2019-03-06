# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:55:04 2016

@author: MindLab
"""

import theano.tensor.signal.downsample as DWS
from tracking.model.theano.Layer import Layer

class MxpLayer(Layer):
    
    def __init__(self, name, knlSize):
        Layer.__init__(self, name, False)
        self.knlSize = knlSize
        
        
    def forward(self, inData):
        fmap = DWS.max_pool_2d(inData, (self.knlSize, self.knlSize), ignore_border=True)
        
        return fmap
        
        
    def getParams(self):
        
        return []
        
        
    def getOutputDim(self, inDims):
        chans, height, width = inDims
        height, width = ((height - self.knlSize) / self.knlSize + 1), ((width - self.knlSize) / self.knlSize + 1)
        outputDim = (chans, height, width)
        
        return outputDim