# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:08:26 2016

@author: MindLAB
"""

import numpy as NP
import numpy.linalg as NLA
import tracking.util.data.Preprocess as Preprocess
from keras.layers import Input
from PIL import Image
from tracking.model.keras.SpatialTransformer import SpatialTransformer
from tracking.sequence.syntetic.model.Object import Object
from tracking.util.model.CentroidPM import CentroidPM

class ObjectGenerator(object):
    
    # imageShape = (channels, height, width)
    def __init__(self, muS, lambdaS, muR, lambdaR, sampler, length, imageShape, trajMod):
        self.samplerS = lambda size : NP.random.laplace(loc=muS, scale=lambdaS, size=size)
        self.samplerR = lambda size : NP.random.laplace(loc=muR, scale=lambdaR, size=size)
        self.sampler = sampler
        self.length = length
        self.imageShape = imageShape
        self.trajMod = trajMod
        
        # Building components
        self.pm = CentroidPM(100, 100)
        self.transformer = self.buildTransformer(imageShape, 1.0)
        
        
    def getLength(self):
        
        return self.length
        
        
    def getImageShape(self):
        
        return self.imageShape
        
    
    def getBatch(self, batchSize):
        batch = []
        theta = self.generateTransformations(batchSize, self.length)
        
        for i in range(batchSize):
            sample = self.sampler.getSample()
            sample.resize(self.imageShape[1:][::-1])
            bbox = sample.getBbox()
            bbox = Preprocess.scalePosition(bbox, sample.getImage().size)
            bbox = self.pm.fromTwoCorners(bbox)
            
            thetaC = NP.zeros((3, 3), dtype='float32')
            thetaC[0, 0] = 1.0
            thetaC[1, 1] = 1.0
            thetaC[[0, 1], [2, 2]] = bbox
            thetaC[2, 2] = 1.0
            
            image = NP.tile(sample.getTensor(), (self.length, 1, 1, 1))
            polygon = NP.tile(sample.getPolygon(), (self.length, 1))
            polygon = Preprocess.scalePosition(polygon, sample.getImage().size)
            image = image.transpose(0, 3, 1, 2)
            thetaI = NP.dot(theta[i, ...], thetaC)
            
            image = self.transformer.predict_on_batch([image, thetaI])
            polygon = self.pm.transform(NLA.inv(thetaI), polygon)
            polygon = Preprocess.rescalePosition(polygon, sample.getImage().size)
            image = image.transpose(0, 2, 3, 1)
            sequence = []
            
            for j in range(self.length):
                obj = Object(Image.fromarray(image[j, ...].astype(NP.uint8)), polygon[j, ...])
                sequence.append(obj)
                
            batch.append(sequence)
            
        return batch
            
    
    def generateTransformations(self, batchSize, length):
        theta = NP.zeros((batchSize, length, 3, 3), dtype='float32')
        position = self.trajMod.getTrajectory(batchSize, length)
        
        # Initialize the trajectory
        theta[:, 0, 0, 0] = 1.0
        theta[:, 0, 1, 1] = 1.0
        theta[:, :, 2, 2] = 1.0
        theta[:, 0, [0, 1], [2, 2]] = position[:, 0, :]
        
        for t in range(1, length):
            thetaT = NP.zeros((batchSize, 3, 3))
            thetaT[:, [0, 1], [0, 1]] = self.samplerS((batchSize, 2))
            thetaT[:, [0, 1], [1, 0]] = 0.0 #self.samplerR((batchSize, 2))
            thetaT[:, [0, 1], [2, 2]] = -position[:, t, :]
            thetaT[:, 2, 2] = 1.0
            theta[:, t, ...] = NP.matmul(theta[:, t-1, ...], thetaT)
            
        return theta
    
    
    def buildTransformer(self, imageShape, downsampleFactor):
        frame = Input(shape=imageShape)
        theta = Input(shape=(3, 3))
        transformer = SpatialTransformer([frame, theta], downsampleFactor).getModel()
        transformer.compile(optimizer="rmsprop", loss='mse')
        
        return transformer