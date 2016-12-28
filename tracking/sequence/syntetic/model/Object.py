# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:47:28 2016

@author: MindLAB
"""

import numpy as NP
import numpy.linalg as NLA
import tracking.util.data.Preprocess as Preprocess
from PIL import Image, ImageDraw
from tracking.util.model.CentroidPM import CentroidPM

class Object(object):
    
    def __init__(self, image, polygon):
        self.image = image
        
        if self.image.mode != "RGB":
            self.image = self.image.convert("RGB")

        self.polygon = NP.array(polygon)
        self.pm = CentroidPM(100, 100)
        
        
    def getPolygon(self):
        
        return self.polygon
    
    
    def getBbox(self):
        position = self.getPolygon()
        positionShape = position.shape
        targetDim = positionShape[-1]
        position = position.reshape((-1, targetDim))
        samples = position.shape[0]
        position = position.reshape(samples, -1, 2)
        bbox = NP.concatenate((NP.min(position, axis=1), NP.max(position, axis=1)), axis=1)
        bbox = bbox.reshape(positionShape[:-1] + (4, ))
        
        return bbox
    
    
    def getImage(self):
        
        return self.image
    
    
    def getTensor(self):
        image = self.getImage()
        image = NP.array(image)
        
        return image
        
        
    # size = (width, height)
    def resize(self, size):
        originalSize = self.image.size # imageSize must be (width, height)
        polygon = Preprocess.scalePosition(self.polygon, originalSize)
        self.polygon = Preprocess.rescalePosition(polygon, size)
        self.image = self.image.resize(size)
        
        
    def getMaskedObject(self):
        mask = Image.new('L', self.image.size, 0)
        maskDraw = ImageDraw.Draw(mask)
        maskDraw.polygon(list(self.getPolygon()), fill=255)
        imageCopy = self.getImage().copy()
        imageCopy.putalpha(mask)
        
        return imageCopy
        
        
    def transform(self, theta):
        image = NP.expand_dims(self.getTensor(), axis=0).transpose(0, 3, 1, 2)
        theta = NP.expand_dims(theta, axis=0)
        image = self.transformer.predict_on_batch([image, theta])[0, ...].transpose(1, 2, 0).astype(NP.uint8)
        image = Image.fromarray(image)
        
        polygon = Preprocess.scalePosition(self.polygon, self.image.size)
        polygon = self.pm.transform(NLA.inv(theta[0, ...]), polygon)
        polygon = Preprocess.rescalePosition(polygon, self.image.size)
        
        return Object(image, polygon)