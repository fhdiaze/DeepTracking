# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:45:58 2016

@author: MindLab
"""

import numpy as NP
import theano as TH
import numpy.linalg as NLA
import tracking.util.data.Preprocess as Preprocess

class Sampler(object):
    
    def __init__(self, transformer, transRange):
        self.transformer = transformer
        self.transRange = transRange
        
        
    def generateSamples(self, frame, position, samples):
        # Generating the transformataions
        frameDims = frame.shape
        frame = frame.transpose(2, 0, 1)
        position = Preprocess.scalePosition(position, frameDims[:2])
        theta, thetaInv = self.generateTheta(position, samples, self.transRange)
        
        # Generating the sampled frames
        frames = TH.shared(frame)
        frames = TH.tensor.tile(frames, (samples, 1, 1, 1))
        output = self.transformer.get_output_for((frames, theta[:, :2, :]))
        frames = output.eval()
        frames = frames.transpose(0, 2, 3, 1)
        
        # Generating the sampled positions
        positions = NP.vstack([NP.ravel(position).reshape((2, -1), order="F"), NP.ones((1, 2))])
        positions = NP.dot(thetaInv, positions)[:, :2, :].T
        positions = NP.ravel(positions).reshape((samples, -1), order="F")
        
        positions = Preprocess.rescalePosition(positions, frameDims[:2])
        
        return frames, positions


    def generateTheta(self, position, samples, transRange):
        thetaC = NP.zeros((samples, 3, 3), dtype='float32')
        objCenter = NP.sum(position.reshape((-1, 2)).T, axis=1, keepdims=True) / 2.0
        tx = objCenter[0,0]
        ty = objCenter[1,0]
        sx = 1.0 - abs(tx) # Scale x
        sy = 1.0 - abs(ty) # Scale y
        
        # Setting transformation
        thetaC[:, 0, 0] = sx
        thetaC[:, 1, 1] = sy
        thetaC[:, 0, 2] = tx
        thetaC[:, 1, 2] = ty
        thetaC[:, 2, 2] = 1.0
        
        thetaT = NP.zeros((samples, 3, 3), dtype='float32')
        tx = NP.random.uniform(-transRange, transRange, size=(samples))
        ty = NP.random.uniform(-transRange, transRange, size=(samples))
        sx = 1.0 - abs(tx) # Scale x
        sy = 1.0 - abs(ty) # Scale y
        
        # Setting transformation
        thetaT[:, 0, 0] = sx
        thetaT[:, 1, 1] = sy
        thetaT[:, 0, 2] = tx
        thetaT[:, 1, 2] = ty
        thetaT[:, 2, 2] = 1.0
        
        theta = NP.matmul(thetaC, thetaT)
        
        return theta, NLA.inv(theta)