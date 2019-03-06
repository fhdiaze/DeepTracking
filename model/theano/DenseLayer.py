# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 11:23:22 2016

@author: MindLab
"""

import numpy as NP
import theano as TH
import theano.tensor as THT
import tracking.util.data.Random as Random
from tracking.model.theano.Layer import Layer


class DenseLayer(Layer):
    
    def __init__(self, name, inputDim, filters):
        Layer.__init__(self, name, True)
        self.filters = filters
        self.W, self.b = self.initParams(inputDim, filters)
        
        
    def forward(self, inData):
        fmap = THT.dot(inData, self.W) + self.b
        
        return fmap
        
        
    def getParams(self):
        
        return [self.W, self.b]
        
        
    def getOutputDim(self, inDims):
        
        return (self.filters)
        
    def initParams(self, inputDim, filters):
        W = TH.shared(Random.glorotUniform((inputDim, filters)), name='W')
        b = TH.shared(NP.zeros((filters, ), dtype=TH.config.floatX), name='b')
        
        return W, b