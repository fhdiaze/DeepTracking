# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:39:34 2016

@author: MindLAB
"""

import numpy as NP

class Generator(object):
    
    def __init__(self, objGen, backGen):
        self.objGen = objGen
        self.backGen = backGen
        
    
    def getBatch(self, batchSize):
        length = self.objGen.getLength()
        imageShape = self.objGen.getImageShape()
        frameShape = imageShape[-2:] + imageShape[:1]
        frame = NP.empty((batchSize, length) + frameShape)
        position = NP.empty((batchSize, length, 4))
        back = self.backGen.getBatch(batchSize)
        obj = self.objGen.getBatch(batchSize)
        
        for i in range(batchSize):
            for j in range(length):
                o = obj[i][j]
                b = back[i][j]
                crop = o.getMaskedObject()
                b.getImage().paste(crop, None, crop)
                frame[i, j, ...] = NP.array(b.getTensor())
                position[i, j, ...] = o.getBbox()
                
        return frame, position
                
        
                