# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:01:40 2016

@author: MindLab
"""

import numpy as NP
from tracking.model.core.Processor import Processor

class ImagenetProcessor(Processor):
    
    def __init__(self, positionModel, tRange):
        self.positionModel = positionModel        
        self.tRange = tRange
        self.mean = NP.array([103.939, 116.779, 123.68])[NP.newaxis, NP.newaxis, :, NP.newaxis, NP.newaxis] # Mean in BGR
    
    def preprocess(self, frame, position):
        frame = NP.copy(frame).transpose(0, 1, 4, 2, 3)
        frame = frame[:,:,::-1,:,:] # Make BGR
        frame = frame - self.mean
        frameDims = frame.shape[-2:][::-1]
        
        oRange = NP.array([[0.0, 0.0], frameDims]).T
        position = self.positionModel.scale(position, oRange, self.tRange)
        position = self.positionModel.fromTwoCorners(position)

        return frame, position

    
    def postprocess(self, frame, position):
        frame = NP.copy(frame) + self.mean
        frame = frame[:,:,::-1,:,:] # Make RGB
        frameDims = frame.shape[-2:][::-1]
        frame = frame.transpose(0, 1, 3, 4, 2)
        
        oRange = NP.array([[0.0, 0.0], frameDims]).T
        position = self.positionModel.toTwoCorners(position)
        position = self.positionModel.scale(position, self.tRange, oRange)
        
        return frame, position