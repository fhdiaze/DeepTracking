# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:01:40 2016

@author: MindLab
"""

import numpy as NP
import tracking.util.data.Preprocess as Preprocess
from tracking.model.core.Processor import Processor

class VggProcessor(Processor):
    
    def __init__(self, frameDims, positionModel):
        self.frameDims = frameDims
        self.positionModel = positionModel
        self.mean = NP.array([103.939, 116.779, 123.68])[NP.newaxis, NP.newaxis, :, NP.newaxis, NP.newaxis]
    
    def preprocess(self, frame, position):
        frame = frame.astype(float).transpose(0, 1, 4, 2, 3)
        frame = frame[:,:,::-1,:,:] # Make BGR
        frame -= self.mean
        
        position = Preprocess.scalePosition(position, self.frameDims)
        position = self.positionModel.fromTwoCorners(position)

        return frame, position

    
    def postprocess(self, frame, position):
        frame += self.mean
        frame = frame[:,:,::-1,:,:] # Make RGB
        frame = frame.transpose(0, 1, 3, 4, 2)
        
        position = self.positionModel.toTwoCorners(position)
        position = Preprocess.rescalePosition(position, self.frameDims)

        return frame, position