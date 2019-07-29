# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 18:54:48 2016

@author: MindLAB
"""

import cPickle as Pickle
import numpy as NP
import numpy.linalg as NLA
import os as OS
import tracking.util.data.Preprocess as Preprocess
import random as RND
import scipy.misc as SCPM
import time as TM
from keras.layers import Input
from PIL import Image
from tracking.model.keras.SpatialTransformer import SpatialTransformer
from tracking.util.model.TwoCornersPM import TwoCornersPM

class TransformerGenerator(object):
    
    def __init__(self, frameShape, framesPath, seqLength, trajectoryModel, summaryPath):
        self.frameShape = frameShape
        self.framesPath = framesPath
        self.seqLength = seqLength
        self.trajectoryModel = trajectoryModel
        self.transformer = self.buildTransformer(frameShape, 1.0)
        self.summary = self.loadSummary(summaryPath)
        self.randGen = self.initRandomGenerator()
        self.summaryKey = "summary"
        self.pm = TwoCornersPM()
        
        
    def initRandomGenerator(self):
        r = RND.Random()
        r.jumpahead(long(TM.time()))
        
        return r
    
    def getBatch(self, batchSize):
        batchF = NP.empty((batchSize, self.seqLength) + self.frameShape)
        batchP = NP.empty((batchSize, self.seqLength, 4))
        
        for i in range(batchSize):
            frame, position = self.getSample()
            
            # Preprocessing the data
            frame = frame.transpose(2, 0, 1)
            position = Preprocess.scalePosition(position, self.frameShape[1:])
            
            # Executing the sequence generation
            theta = self.trajectoryModel.generateTrajectory(self.seqLength)
            frame = NP.tile(frame, (self.seqLength, 1, 1, 1))
            frame = self.transformer.predict_on_batch([frame, theta])
    
            batchF[i, ...] = frame
            batchP[i, ...] = self.pm.transform(NLA.inv(theta), NP.tile(position, (self.seqLength, 1)))
            
        batchF = batchF.transpose(0, 1, 3, 4, 2)
        batchP = Preprocess.rescalePosition(batchP, self.frameShape[1:])
            
        return batchF, batchP
    
    
    def buildTransformer(self, frameShape, downsampleFactor):
        frame = Input(shape=frameShape)
        theta = Input(shape=(3, 3))
        transformer = SpatialTransformer([frame, theta], downsampleFactor).getModel()
        transformer.compile(optimizer="rmsprop", loss='mse')
        
        return transformer
        
        
    def getSample(self):
        objData = self.randGen.choice(self.summary[self.summaryKey])
        objPath = OS.path.join(self.framesPath, objData['file_name'].strip())
        frame = Image.open(objPath)
        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        frame = NP.array(frame)
        originalSize = frame.shape[:2][::-1] # imageSize must be (width, height)
        position = Preprocess.scalePosition(NP.array(objData["bbox"]), originalSize)
        position = Preprocess.rescalePosition(position, self.frameShape[1:])
        frame = SCPM.imresize(frame, self.frameShape[1:3])
        
        
        return frame, position
        
        
    def loadSummary(self, summaryPath):
        
        with open(summaryPath, 'r') as summaryFile:
            summary = Pickle.load(summaryFile)
            
        return summary