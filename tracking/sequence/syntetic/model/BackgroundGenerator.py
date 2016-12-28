# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:52:00 2016

@author: MindLAB
"""

from tracking.sequence.syntetic.model.Background import Background

class BackgroundGenerator(object):
    
    def __init__(self, sampler, length, imageShape):
        self.sampler = sampler
        self.length = length
        self.imageShape = imageShape
        
    
    def getLength(self):
        
        return self.length
        
        
    def getImageShape(self):
        
        return self.imageShape
        
        
    def getBatch(self, batchSize):
        batch = []
        
        for i in range(batchSize):
            sample = self.sampler.getSample()
            sample.resize(self.imageShape[1:][::-1])
            sequence = []
            for j in range(self.length):
                back = Background(sample.getImage().copy())
                sequence.append(back)
                
            batch.append(sequence)
                
        return batch