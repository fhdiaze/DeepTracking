# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:59:29 2016

@author: MindLab
"""

import numpy as NP
import os
import random
import tracking.data.VotTool as VotTool

class VotGenerator(object):

    # frameShape = (channels, height, width)
    def __init__(self, seqLength, seqsPath, seqs, extension, boxesFile, annsFile):
        self.seqLength = seqLength
        self.seqsPath = seqsPath
        self.seqs = seqs
        self.extension = extension
        self.boxesFile = boxesFile
        self.annsFile = annsFile

	
    def getBatch(self, batchSize):
        batchF = []
        batchP = []

        for i in range(batchSize):
            seqPath = os.path.join(self.seqsPath, random.choice(self.seqs))
            frame, position = self.loadSequence(seqPath, self.extension, self.boxesFile)
            
            batchF.append(frame)
            batchP.append(position)
            
        return batchF, batchP
    
    
    def loadSequence(self, path, extension, boxesFile):
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
        frames = VotTool.loadFrames(framePaths)
        
        # Load bounding boxes information
        boxesPath = os.path.join(path, boxesFile)
        position = VotTool.loadPosition(boxesPath)[start:start+self.seqLength, ...]
        
        return frames, position
    

