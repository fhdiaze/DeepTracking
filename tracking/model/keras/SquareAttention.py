# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 20:06:51 2016

@author: MindLAB
"""

import theano.tensor as THT
from keras import backend as K
from keras.layers import merge
from keras.models import Model
from tracking.model.keras.Module import Module
from tracking.util.theano.Data import Data

class SquareAttention(Module):
    
    def __init__(self, input, alpha, scale):
        if type(input) is not list or len(input) != 2:
            raise Exception("SquareAttention must be called on a list of two tensors. Got: " + str(input))
        
        self.alpha = alpha
        self.scale = scale
        output = merge(input, mode=self.call, output_shape=self.outputShape)
        self.model = Model(input=input, output=output)


    def getModel(self):
        
        return self.model

    
    def call(self, X):
        if type(X) is not list or len(X) != 2:
            raise Exception("SquareAttention must be called on a list of two tensors. Got: " + str(X))
            
        frame, position  = X[0], X[1]
        
        # Reshaping the input to exclude the time dimension
        frameShape = K.shape(frame)
        positionShape = K.shape(position)
        (chans, height, width) = frameShape[-3:]
        targetDim = positionShape[-1]
        frame = K.reshape(frame, (-1, chans, height, width))
        position = K.reshape(position, (-1, ) + (targetDim, ))
        
        # Applying the attention
        hw = THT.abs_(position[:, 2] - position[:, 0]) * self.scale / 2.0
        hh = THT.abs_(position[:, 3] - position[:, 1]) * self.scale / 2.0
        position = THT.maximum(THT.set_subtensor(position[:, 0], position[:, 0] - hw), -1.0)
        position = THT.minimum(THT.set_subtensor(position[:, 2], position[:, 2] + hw), 1.0)
        position = THT.maximum(THT.set_subtensor(position[:, 1], position[:, 1] - hh), -1.0)
        position = THT.minimum(THT.set_subtensor(position[:, 3], position[:, 3] + hh), 1.0)
        rX = Data.linspace(-1.0, 1.0, width)
        rY = Data.linspace(-1.0, 1.0, height)
        FX = THT.gt(rX, position[:,0].dimshuffle(0,'x')) * THT.le(rX, position[:,2].dimshuffle(0,'x'))
        FY = THT.gt(rY, position[:,1].dimshuffle(0,'x')) * THT.le(rY, position[:,3].dimshuffle(0,'x'))
        m = FY.dimshuffle(0, 1, 'x') * FX.dimshuffle(0, 'x', 1)
        m = m + self.alpha - THT.gt(m, 0.) * self.alpha
        frame = frame * m.dimshuffle(0, 'x', 1, 2)
        
        # Reshaping the frame to include time dimension
        output = K.reshape(frame, frameShape)
        
        return output


    def outputShape(self, inputShapes):
        frameShape = inputShapes[0]
        
        return frameShape  