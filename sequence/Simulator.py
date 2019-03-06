# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:56:20 2016

@author: fhdiaze
"""

import numpy as NP

class Simulator:
    # scene PIL.Image
    def __init__(self, scene, obj, polygon, trajectoryModel):
        self.scene = scene
        self.obj = obj
        self.polygon = polygon
        self.trajectoryModel = trajectoryModel
    
    def getSequence(self, length, height, width):
        # frames.shape = (length, channels, height, width)
        frames = NP.empty((0, 3, height, width))
        
        # trajectory.shape = (xCenter, yCenter) x frame
        trajectory, scales = self.trajectoryModel.getTrajectory(length)
        
        for j, frame in enumerate(trajectory):
            data[i, j, :, :, :] = np.asarray(frame.convert('RGB'))
            label[i, j] = simulator.getBox()
        
        