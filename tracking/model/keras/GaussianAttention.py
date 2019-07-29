# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:23:17 2016

@author: MindLAB
"""

from keras import backend as K
from keras.layers import merge
from keras.models import Model
from tracking.model.keras.Module import Module
from tracking.util.theano.Data import Data

class GaussianAttention(Module):
    
    def __init__(self, input, epsilon, alpha):
        if type(input) is not list or len(input) != 2:
            raise Exception("GaussianAttention must be called on a list of two tensors. Got: " + str(input))
        
        self.epsilon = epsilon
        self.alpha = alpha
        output = merge(input, mode=self.call, output_shape=self.outputShape)
        self.model = Model(input=input, output=output)
    
    
    def call(self, X):
        if type(X) is not list or len(X) != 2:
            raise Exception("GaussianAttention must be called on a list of two tensors. Got: " + str(X))
        
        frame, position  = X[0], X[1]
        
        # Reshaping the input to exclude the time dimension
        frameShape = K.shape(frame)
        positionShape = K.shape(position)
        (chans, height, width) = frameShape[-3:]
        targetDim = positionShape[-1]
        frame = K.reshape(frame, (-1, chans, height, width))
        position = K.reshape(position, (-1, ) + (targetDim, ))
        
        cx = (position[:, 0] + position[:, 2]) / 2.0
        cy = (position[:, 1] + position[:, 3]) / 2.0
        sx = (position[:, 2] - cx) * 0.60
        sy = (position[:, 3] - cy) * 0.60
        rX = Data.linspace(-1.0, 1.0, width)
        rY = Data.linspace(-1.0, 1.0, height)
        FX = K.exp(-(rX - cx.dimshuffle(0, 'x')) ** 2 / (2.0 * (sx.dimshuffle(0, 'x') ** 2 + self.epsilon)))
        FY = K.exp(-(rY - cy.dimshuffle(0, 'x')) ** 2 / (2.0 * (sy.dimshuffle(0, 'x') ** 2 + self.epsilon)))
        m = (FY.dimshuffle(0, 1, 'x') * FX.dimshuffle(0, 'x', 1))
        m = m + self.alpha
        m = m - K.greater(m, 1.0) * (m - 1.0)
        
        frame = frame * m.dimshuffle(0, 'x', 1, 2)
        
        # Reshaping the frame to include time dimension
        output = K.reshape(frame, frameShape)
        
        return output
        
        
    def outputShape(self, inputShapes):
        frameShape = inputShapes[0]
        
        return frameShape