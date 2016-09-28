# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 22:30:52 2016

@author: MindLAB
"""

#!/usr/bin/python

import logging
import numpy as NP
import os
import scipy.misc as SCPM
import sys
import tracking.util.data.Preprocess as Preprocess
import tracking.util.data.VotTool as VotTool
import vot

stderr_ = sys.stderr
sys.stderr = open(os.devnull, "w")

from tracking.model.keras.Tracker import Tracker
from tracking.util.data.GeneralProcessor import GeneralProcessor
from tracking.util.model.TwoCornersPM import TwoCornersPM

sys.stderr.close()
sys.stderr = stderr_

def loadTracker(path="/home/fhdiaze/Models/Tracker22A.pkl"):
    tracker = Tracker()
    tracker.load(path)
    
    return tracker


def track(tracker, processor, frame, position, size):

    x, y, w, h = position
    x1, y1 = x + w, y + h

    position = NP.array([x, y, x1, y1])
    position = NP.expand_dims(position, axis=0)
    position = NP.expand_dims(position, axis=1)
    
    originalSize = frame.shape[:2][::-1] # imageSize must be (width, height)
    frame = SCPM.imresize(frame, size)
    frame = NP.expand_dims(frame, axis=0)
    frame = NP.expand_dims(frame, axis=1)
    
    position = Preprocess.scalePosition(position, originalSize)
    position = Preprocess.rescalePosition(position, size)

    frame, position = processor.preprocess(frame, position)
    position = tracker.forward([frame], position[:, 0, :])
    
    x, y, x1, y1 = position[0, 0, :]
    
    #logging.info("Tracker prediction: [%s, %s, %s, %s]", x, y, x1, y1)
    
    return vot.Rectangle(x, y, x1 - x, y1 - y)



# LOGGING VARIABLES
trackerName = "Tracker22AVot"
logFilePath = "/home/fhdiaze/Data/Log/" + trackerName + ".log"
logFormat = "%(asctime)s:%(levelname)s:%(funcName)s:%(lineno)d:%(message)s"
logDateFormat = "%H:%M:%S"
logLevel = logging.INFO

logger = logging.getLogger()
fileHandler = logging.FileHandler(logFilePath, mode='w')
formatter = logging.Formatter(logFormat)
fileHandler.setFormatter(formatter)
logger.handlers = []
logger.addHandler(fileHandler)
logger.setLevel(logLevel)

# *****************************************
# VOT: Create VOT handle at the beginning
#      Then get the initializaton region
#      and the first image
# *****************************************
handle = vot.VOT("rectangle")
position = handle.region()

# Process the first frame
framePath = handle.frame()

if not framePath:
    sys.exit(0)
    
frameDims = (3, 224, 224) # (channels, width, height)
positionRepresentation = TwoCornersPM()
processor = GeneralProcessor(frameDims[1:3], positionRepresentation)
frame = VotTool.loadFrame(framePath)
tracker = loadTracker()
logging.info("Tracker was loaded: %s", tracker.loss)

while True:
    # *****************************************
    # VOT: Call frame method to get path of the 
    #      current image frame. If the result is
    #      null, the sequence is over.
    # *****************************************
    framePath = handle.frame()
    if not framePath:
        break
    
    frame = VotTool.loadFrame(framePath)
    position = track(tracker, processor, frame, position, frameDims[1:3])
    
    # *****************************************
    # VOT: Report the position of the object 
    #      every frame using report method.
    # *****************************************
    handle.report(position)