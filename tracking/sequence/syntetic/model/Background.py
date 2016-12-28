# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:28:46 2016

@author: MindLAB
"""

import numpy as NP

class Background(object):
    
    def __init__(self, image):
        self.image = image
        
        if self.image.mode != "RGB":
            self.image = self.image.convert("RGB")
        
        
    def getImage(self):
        
        return self.image
    
    
    def getTensor(self):
        image = self.getImage()
        image = NP.array(image)
        
        return image
        
        
    # size = (width, height)
    def resize(self, size):
        self.image = self.image.resize(size)