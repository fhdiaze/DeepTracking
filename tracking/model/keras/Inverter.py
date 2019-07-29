# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 13:34:48 2016

@author: MindLAB
"""

import theano.tensor.nlinalg as TLA
from keras.layers.core import Lambda
from keras.models import Model
from keras import backend as K
from tracking.model.keras.Module import Module

class Inverter(Module):
    
    def __init__(self, input):
        self.build(input)
        
        
    def build(self, input):
        invert = Lambda(self.call, output_shape=self.outputShape)
        output = invert(input)
        self.model = Model(input=input, output=output)
        
        
    def step(self, X, states):
        inv = TLA.matrix_inverse(X)
        
        return inv, []

    
    def call(self, X):
        xShape = K.shape(X)
        X = K.reshape(X, (-1, 3, 3))
        X = X.dimshuffle(1, 0, 2)
        lastOut, outputs, states = K.rnn(self.step, X, initial_states=[])
        outputs = outputs.dimshuffle(1, 0, 2)
        outputs = K.reshape(outputs, xShape)
        
        return outputs

        
    def outputShape(self, inputShape):
        
        return inputShape