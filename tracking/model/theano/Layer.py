# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 20:13:37 2016

@author: MindLab
"""

class Layer(object):
    
    def __init__(self, name, trainable):
        self.name = name
        self.trainable = trainable
        
        
    def forward(self, inData):
        pass
    
    
    def setTrainable(self, trainable):
        self.trainable = trainable