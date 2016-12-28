# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 22:07:41 2016

@author: MindLab
"""

import numpy as NP
from PIL import Image

def scaleFrame(frame):
    
    return (frame - 127.) / 127.
    

def rescaleFrame(frame):
    
    return frame * 127.0 + 127.0
    

# size = (width, height)
def scalePosition(position, size):
    shape = position.shape
    scaledPosition = NP.ravel(position).reshape((-1, 2)).T / (NP.array([size]).T / 2.0) - 1.0
    scaledPosition = NP.ravel(scaledPosition, order="F").reshape(shape)
    
    return scaledPosition


# size = (height, width)
def rescalePosition(position, size):
    shape = position.shape
    rescaledPosition = (NP.ravel(position).reshape((-1, 2)).T + 1.0) * (NP.array([size]).T / 2.0)
    rescaledPosition = NP.ravel(rescaledPosition, order="F").reshape(shape)
    
    return rescaledPosition
    
    
# frame.shape = (batchSize, seqLength, channels, height, width)    
def rgbToGray(frame):
    pass


"""
Scale the bounding boxes from a frame dimensions to another

@type bboxes: numpy.array(frames, coordinates)
@param bboxes: The array with the bounding boxes, one per row
@type fromSize: [height, width]
@param fromSize: The dimensions of the source frames
@type toSize: [height, width]
@param toSize: The dimensions of the target frames

@rtype: numpy.array(frames, coordinates)
@return: The array with the bounding boxes, one per row
"""
def scaleBboxes(bboxes, fromSize, toSize):
    scaledFromSize = 1.0 / NP.array(fromSize)
    coords = bboxes.shape[1]
    scaledFromSize = NP.multiply(scaledFromSize, NP.ones((2, coords / 2))).flatten()
    scaledBboxes = NP.multiply(bboxes, scaledFromSize)
    toScaledSize = NP.multiply(toSize, NP.ones((2, coords / 2))).flatten()
    return NP.multiply(scaledBboxes, toScaledSize)
    

def resizeSequence(frames, position, newSize):
    currentSize = frames[0].size
    newFrames = []
    
    for frame in frames:
        frame = frame.resize(newSize)
        newFrames.append(frame)
        
    position = scalePosition(position, currentSize)
    position = rescalePosition(position, newSize)
    
    return newFrames, position
    
    
# image = (chans, h, w)
def getFilters(qty, image, position, knlSize):
    chans, heigth, width = image.shape
    filters = NP.zeros((qty, chans, knlSize, knlSize))
    x1, y1, x2, y2 = position
    xMax = x2 - x1 - knlSize 
    yMax = y2 - y1 - knlSize

    for i in range(qty):
        x = x1 + NP.random.randint(0, xMax)
        y = y1 + NP.random.randint(0, yMax)
        filters[i, ...] = image[:, y:y+knlSize, x:x+knlSize]
        
    return filters
    
    
# Convert tensor of frames to an iterator of images
def getFrames(frames):
    fs, _, _, _ = frames.shape
    
    for i in range(fs):
        image = Image.fromarray(frames[i, ...].astype(NP.uint8))
        
        yield image