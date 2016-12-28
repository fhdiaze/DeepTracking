# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:12:33 2016

@author: MindLAB
"""

import numpy as NP
import tracking.util.data.Preprocess as Preprocess
from tracking.model.core.Processor import Processor

class GeneralProcessor(Processor):
    
    def __init__(self, positionModel, tRange):
        self.positionModel = positionModel
        self.tRange = tRange
    
    
    def preprocess(self, frame, position):
        frame = NP.copy(frame).transpose(0, 1, 4, 2, 3)
        #frame = frame[:,:,::-1,:,:] # Make BGR
        frame = Preprocess.scaleFrame(frame)
        frameDims = frame.shape[-2:][::-1]
        
        oRange = NP.array([[0.0, 0.0], frameDims]).T
        position = self.positionModel.scale(position, oRange, self.tRange)
        position = self.positionModel.fromTwoCorners(position)

        return frame, position

    
    def postprocess(self, frame, position):
        frame = Preprocess.rescaleFrame(NP.copy(frame))
        #frame = frame[:,:,::-1,:,:] # Make RGB
        frameDims = frame.shape[-2:][::-1]
        frame = frame.transpose(0, 1, 3, 4, 2)
        
        oRange = NP.array([[0.0, 0.0], frameDims]).T
        position = self.positionModel.toTwoCorners(position)
        position = self.positionModel.scale(position, self.tRange, oRange)

        return frame, position