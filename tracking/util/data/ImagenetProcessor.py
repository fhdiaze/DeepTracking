# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 19:01:40 2016

@author: MindLab
"""

import numpy as NP
import tracking.util.data.Preprocess as Preprocess
from tracking.model.core.Processor import Processor

class ImagenetProcessor(Processor):
    
    def __init__(self, positionModel):
        self.positionModel = positionModel
        self.mean = NP.array([103.939, 116.779, 123.68])[NP.newaxis, NP.newaxis, :, NP.newaxis, NP.newaxis] # Mean in BGR
    
    def preprocess(self, frame, position):
        frame = NP.copy(frame).transpose(0, 1, 4, 2, 3)
        frame = frame[:,:,::-1,:,:] # Make BGR
        frame = frame - self.mean
        frameDims = frame.shape[-2:]
        
        position = Preprocess.scalePosition(position, frameDims)
        position = self.positionModel.fromTwoCorners(position)

        return frame, position

    
    def postprocess(self, frame, position):
        frame = NP.copy(frame) + self.mean
        frame = frame[:,:,::-1,:,:] # Make RGB
        frameDims = frame.shape[-2:]
        frame = frame.transpose(0, 1, 3, 4, 2)
        
        position = self.positionModel.toTwoCorners(position)
        position = Preprocess.rescalePosition(position, frameDims)
        
        return frame, position