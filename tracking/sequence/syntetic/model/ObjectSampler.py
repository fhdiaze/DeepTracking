# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:20:23 2016

@author: MindLAB
"""

import cPickle as Pickle
import os as OS
import random as RND
import time as TM
from PIL import Image
from tracking.sequence.syntetic.model.Object import Object

class ObjectSampler(object):
    
    def __init__(self, objectSetPath, basePath):
        self.objectSet = self.loadObjectSet(objectSetPath)
        self.basePath = basePath
        self.key = "summary"
        self.randGen = self.initRandomGenerator()
    
    
    def getSample(self):
        objData = self.randGen.choice(self.objectSet[self.key])
        imagePath = OS.path.join(self.basePath, objData['file_name'].strip())
        polygon = objData["segmentation"][0]
        image = Image.open(imagePath)
        
        return Object(image, polygon)
        
        
    def loadObjectSet(self, path):
        
        with open(path, 'r') as objectSetFile:
            objectSet = Pickle.load(objectSetFile)
            
        return objectSet
    
    
    def initRandomGenerator(self):
        r = RND.Random()
        r.jumpahead(long(TM.time()))
        
        return r