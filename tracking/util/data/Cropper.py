# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 14:31:07 2016

@author: MindLAB
"""

import numpy as NP
import numpy.linalg as NLA
from PIL import Image
from tracking.data.Processor import Processor
from tracking.model.core.CentroidHWPM import CentroidHWPM


class Cropper(Processor):
    
    def __init__(self, positionModel, context, distorsion, minSide, tRange, frameSize):
        self.positionModel = positionModel
        self.context = context
        self.distorsion = distorsion
        self.minSide = minSide
        self.tRange = tRange
        self.frameSize = frameSize
        self.chwPM = CentroidHWPM()
        self.cRange = NP.array([[-1.0, 1.0], [-1.0, 1.0]])
        
    
    # cropPosition.shape = (batchSize, seqLength, targetDim)
    # objPosition.shape = (batchSize, seqLength, targetDim)
    # frame = [[]]
    def crop(self, frame, cropPosition, objPosition):        
        frameSize = frame[0].size[0]
        
        # Generating the frames crops
        frame = self.cropFrames(frame, cropPosition)
        oRange = NP.array([[0, frameSize], [0, frameSize]])
        
        # Generating the positions
        
        # Generating the transformations
        cropPosition = self.positionModel.scale(cropPosition, oRange, self.cRange)
        theta, thetaInv = self.generateTheta(cropPosition)
        objPosition = self.positionModel.scale(objPosition, oRange, self.cRange)
        objPosition = self.positionModel.transform(thetaInv, objPosition)
        objPosition = self.positionModel.scale(objPosition, self.cRange, self.tRange)
        
        return frame, objPosition, theta
        
        
    def generateTheta(self, position):
        seqLength, targetDim = position.shape
        samples = seqLength
        theta = NP.zeros((samples, 3, 3), dtype='float32')
        
        targetDim = self.chwPM.getTargetDim()
        chw = self.chwPM.fromTwoCorners(position)
        chw = chw.reshape((samples, targetDim))
        dx = NP.random.uniform(-self.distorsion, self.distorsion, size=(samples))
        dy = NP.random.uniform(-self.distorsion, self.distorsion, size=(samples))
        cX = chw[:, 0] + dx
        cY = chw[:, 1] + dy
        h = NP.maximum(chw[:, 2] * self.context, self.minSide)
        w = NP.maximum(chw[:, 3] * self.context, self.minSide)
        
        # Calculating the parameters of the transformation
        tx = cX
        ty = cY
        sx = w / 2.0 # Scale x
        sy = h / 2.0 # Scale y
        
        # Setting transformation
        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        theta[:, 2, 2] = 1.0
        
        return theta, NLA.inv(theta)
        
        
    def cropFrames(self, frame, box):
        
        length = len(frame)
        _, _, channels = NP.asarray(frame[0]).shape
        window = self.chwPM.fromTwoCorners(box)
        window[:,  2:] *= self.context
        window = self.chwPM.toTwoCorners(window)
        
        cFrame = NP.zeros((length, self.frameSize, self.frameSize, channels))
        
        for t in range(length):
            f = frame[t]
            f = f.crop(window[t, ...])
            f = f.resize((self.frameSize, self.frameSize), Image.BICUBIC)
            cFrame[t, ...] = NP.asarray(f)
        
        return cFrame