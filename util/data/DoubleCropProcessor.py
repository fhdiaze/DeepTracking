# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:54:20 2016

@author: MindLAB
"""

class DoubleCropProcessor(object):
    
    def __init__(self, processor, cropper, transformer):
        self.processor = processor
        self.cropper = cropper
        self.transformer = transformer
        
    
    def preprocess(self, frame, position):
        frame, position = self.processor.preprocess(frame, position)
        
        if position.shape[1] == 1:
            objP = position[:, 0:, :]
            regP = position[:, 0:, :]
        else:
            objP = position[:, :-1, :]
            regP = position[:, 1:, :]
        
        objF, _, objT = self.cropper.crop(frame[:, :-1, ...], objP, objP)
        regF, regP, self.regT = self.cropper.crop(frame[:, 1:, ...], objP, regP)
        
        return [objF, regF], regP
    
    
    def postprocess(self, frame, position):
        position = self.transformer.transform(self.regT, position)
        frame, position = self.processor.postprocess(frame[0], position)
        
        return frame, position