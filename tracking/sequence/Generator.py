# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:58:53 2016

@author: MindLAB
"""

from GaussianGenerator import GaussianGenerator

class Generator(object):
    
    def __init__(self, imgSize, objectPathTemplate, scenePathTemplate,seqLength,
                 imageDir, summaryPath, trajectoryModelPath, trajectoryModelSpec, cameraTrajectoryModelSpec):
        self.seqLength = seqLength
        
        grayscale = False
        parallel = True
        numProcs = 2
        flow = False
        self.generator = GaussianGenerator(imageDir, summaryPath, trajectoryModelSpec, 
                            cameraTrajectoryModelSpec, trajectoryModelPath, self.seqLength, imageSize=imgSize[0],
                            grayscale=grayscale, parallel=parallel, numProcs=numProcs, computeFlow=flow,
                            scenePathTemplate=scenePathTemplate, objectPathTemplate=objectPathTemplate)
    
    
    def getBatch(self, batchSize):
        frame, position, _ = self.generator.getBatch(batchSize, 0, self.seqLength)
        
        return frame, position