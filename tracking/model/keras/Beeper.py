# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:21:15 2016

@author: MindLAB
"""

import theano.tensor as THT
import theano.tensor.extra_ops as THEO
from keras import backend as K
from keras.layers.core import Lambda
from keras.models import Model
from theano.tensor.shared_randomstreams import RandomStreams
from tracking.model.keras.Module import Module

class Beeper(Module):
    
    def __init__(self, input, distortion, minSide, context):
        self.distortion = distortion
        self.minSide = minSide
        self.context = context
        self.srng = RandomStreams()
        self.build(input)
        

    def build(self, input):
        beeper = Lambda(self.call, output_shape=self.getOutputShape)
        output = beeper(input)
        self.model = Model(input=input, output=output)
        
        
    def getModel(self):
        
        return self.model


    def call(self, position):
        inputDim = K.ndim(position)
        positionShape = K.shape(position)
        targetDim = positionShape[-1]
        position = K.reshape(position, (-1, targetDim))
        samples = K.shape(position)[0]
        theta = THT.zeros((samples, 3, 3))
        
        chw = self.toChw(position)
        chw = K.reshape(chw, (samples, targetDim))
        dx = -self.distortion + 2.0 * self.distortion * self.srng.uniform((samples,)) 
        dy = -self.distortion + 2.0 * self.distortion * self.srng.uniform((samples,))
        cX = chw[:, 0] + dx
        cY = chw[:, 1] + dy
        h = K.maximum(chw[:, 2] * (1.0 + self.context), self.minSide)
        w = K.maximum(chw[:, 3] * (1.0 + self.context), self.minSide)
        
        # Calculating the parameters of the transformation
        tx = cX
        ty = cY
        sx = w / 2.0 # Scale x
        sy = h / 2.0 # Scale y
        
        # Setting transformation
        theta = THT.set_subtensor(theta[:, 0, 0], sx)
        theta = THT.set_subtensor(theta[:, 1, 1], sy)
        theta = THT.set_subtensor(theta[:, 0, 2], tx)
        theta = THT.set_subtensor(theta[:, 1, 2], ty)
        theta = THT.set_subtensor(theta[:, 2, 2], 1.0)
        
        thetaShape = K.concatenate([positionShape[:-1], K.shape(theta)[-2:]])
        theta = THT.reshape(theta, thetaShape, ndim=inputDim + 1)
        
        return theta


    def getOutputShape(self, inputShapes):
        outputShape = inputShapes[:-1] + (3, 3)
        
        return outputShape
        
        
    def toChw(self, position):
        samples, targetDim = K.shape(position)
        position = K.reshape(position, (samples, 2, 2))
        centroid = K.sum(position, axis=1) / 2.0
        hw = K.abs(THEO.diff(position, axis=1)[:,0,:])
        chw = K.concatenate((centroid, hw), axis=1)
        chw = chw[:, [0, 1, 3, 2]] # Changing from cwh to chw
        
        return chw