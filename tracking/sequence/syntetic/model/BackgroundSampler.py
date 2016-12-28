# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:54:32 2016

@author: MindLAB
"""

import os
import random as RND
import time as TM
from PIL import Image
from tracking.sequence.syntetic.model.Background import Background

class BackgroundSampler(object):
    
    def __init__(self, baseBackPath):
        self.baseBackPath = baseBackPath
        self.backSet = os.listdir(baseBackPath)
        self.randGen = self.initRandomGenerator()
        
        
    def getSample(self):
        backName = self.randGen.choice(self.backSet)
        backPath = os.path.join(self.baseBackPath, backName)
        image = Image.open(backPath)
        
        return Background(image)
        
        
    def initRandomGenerator(self):
        r = RND.Random()
        r.jumpahead(long(TM.time()))
        
        return r