# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:59:29 2016

@author: MindLab
"""

import numpy as NP
import os
import random
import tracking.util.data.Preprocess as Preprocess
import tracking.util.data.VotTool as VotTool

class VotGenerator(object):

    # frameShape = (channels, height, width)
    def __init__(self, frameShape, seqLength, seqsPath, seqs, extension, boxesFile, annsFile):
        self.frameShape = frameShape[-2:] + frameShape[:1]
        self.seqLength = seqLength
        self.seqsPath = seqsPath
        self.seqs = seqs
        self.extension = extension
        self.boxesFile = boxesFile
        self.annsFile = annsFile

	
    def getBatch(self, batchSize):
        batchF = NP.empty((batchSize, self.seqLength) + self.frameShape)
        batchP = NP.empty((batchSize, self.seqLength, 4))

        for i in range(batchSize):
            seqPath = os.path.join(self.seqsPath, random.choice(self.seqs))
            frameSize = self.frameShape[:2][::-1] # imageSize must be (width, height)
            frame, position = self.loadSequence(seqPath, self.extension, self.boxesFile, frameSize)
            
            batchF[i, ...] = frame
            batchP[i, ...] = position
            
        return batchF, batchP
    
    
    def loadSequence(self, path, extension, boxesFile, size):
        # Load frames
        framePaths = VotTool.getFramePaths(path, extension)
        
        if self.annsFile:
            annsPath = os.path.join(path, self.annsFile)
            anns = NP.loadtxt(annsPath, dtype=int)
            minF = NP.min(anns)
            maxF = NP.max(anns)
        else:
            minF = 0
            maxF = len(framePaths)
        
        start = NP.random.randint(minF, maxF - self.seqLength)
        framePaths = framePaths[start:start+self.seqLength]
        frame, originalSize = VotTool.loadFrames(framePaths, size)
        
        # Load bounding boxes information
        boxesPath = os.path.join(path, boxesFile)
        position = VotTool.loadPosition(boxesPath)[start:start+self.seqLength, ...]
        position = Preprocess.scalePosition(position, originalSize)
        position = Preprocess.rescalePosition(position, size)
        
        return frame, position
    

