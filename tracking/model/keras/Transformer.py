# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 21:24:21 2016

@author: MindLAB
"""

from keras import backend as K
from keras.layers import merge
from keras.models import Model
from theano import tensor as T
from tracking.model.keras.Module import Module

class Transformer(Module):
    
    def __init__(self, input):
        if type(input) is not list or len(input) != 2:
            raise Exception("Transformer must be called on a list of two tensors. Got: " + str(input))
        
        output = merge(input, mode=self.call, output_shape=self.outputShape)
        self.model = Model(input=input, output=output)


    def getModel(self):
        
        return self.model


    def call(self, X):
        if type(X) is not list or len(X) != 2:
            raise Exception("Transformer must be called on a list of two tensors. Got: " + str(X))
                            
        position, theta = X[0], X[1]
        positionShape = K.shape(position)
        position = K.reshape(position, (-1, positionShape[-1]))
        theta = theta.reshape((-1, 3, 3))
        output = Transformer.transform(position, theta)
        output = K.reshape(output, positionShape)

        return output


    def outputShape(self, inputShapes):
        positionShape = inputShapes[0]
        
        return positionShape
        
        
    @staticmethod
    def transform(position, theta):
        (batchSize, targetDim) = K.shape(position)
        
        # Reshaping the positions
        position = K.reshape(K.flatten(position), (batchSize, 2, -1))
        position = K.permute_dimensions(position, (0, 2, 1))
        position = K.concatenate([position, T.ones((batchSize, 1, K.shape(position)[2]))], axis=1)
        
        # Applying the transformation
        position = K.batch_dot(theta, position)[:, :2, :]
        
        # Reshaping the result
        position = K.permute_dimensions(position, (0, 2, 1))
        position = K.reshape(position, (batchSize, targetDim))
        
        return position