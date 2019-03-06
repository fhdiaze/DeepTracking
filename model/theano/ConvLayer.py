# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:20:32 2016

@author: MindLab
"""

import theano as TH
import tracking.util.data.Random as Random
from tracking.model.theano.Layer import Layer

class ConvLayer(Layer):
    
    def __init__(self, name, convEngine, filters, inChans, knlSize, stride, W):
        Layer.__init__(self, name, True)
        self.convEngine = convEngine
        self.filters = filters
        self.knlSize = knlSize
        self.stride = stride
        
        if W is None:
            self.W = self.initConv(filters, inChans, knlSize, name)
        else:
            self.W = TH.shared(W, name=name)
        
        
    def forward(self, inData):
        fmap = self.convEngine(inData, self.W, subsample=(self.stride, self.stride))
        
        return fmap
        
        
    def initConv(self, filters, inChans, knlSize, name):
        return TH.shared(Random.glorotUniform((filters, inChans, knlSize, knlSize)), name=name)
        
        
    def getParams(self):
        
        return [self.W]
        
        
    def getOutputDim(self, inDims):
        chans, height, width = inDims
        height, width = ((height - self.knlSize) / self.stride + 1), ((width - self.knlSize) / self.stride + 1)
        outputDim = (self.filters, height, width)
        
        return outputDim