# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 19:54:40 2016

@author: MindLab
"""

import numpy as NP
import theano.tensor as THT
from tracking.model.theano.Layer import Layer

class Flatten(Layer):
    
    def __init__(self, name):
        Layer.__init__(self, name, False)
        
    def forward(self, inData):
        flat = THT.reshape(inData, (inData.shape[0], THT.prod(inData.shape[1:])))
        
        return flat
        
        
    def getParams(self):
        
        return []
        
        
    def getOutputDim(self, inDims):
        
        return (NP.prod(inDims))