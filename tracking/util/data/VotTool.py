# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:59:29 2016

@author: MindLab
"""

import os
import random
import numpy as NP
import dill
import matplotlib.pyplot as PLT
import tracking.data.Preprocess as Preprocess
from PIL import Image
from tracking.data.Sequence import Sequence


def loadFrame(path):
    frame = Image.open(path)
    frame.load()
    
    if frame.mode != "RGB":
            frame = frame.convert("RGB")
    
    return frame
    
    
def loadFrames(framePaths):
    frames = []
    
    for i, framePath in enumerate(framePaths):
        tmpFrame = loadFrame(framePath)
        frames.append(tmpFrame)
    
    return frames
    

def getFramePaths(path, extension):
    framePaths = [os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(extension)]
    framePaths = sorted(framePaths)
    
    return framePaths
        
        
def loadPosition(path):
    # Load bounding boxes information
    position = []
    
    with open(path) as f:
        [position.append(list(map(float, line.split(",")))) for line in f]
    
    position = NP.array(position)
    position = position.reshape(-1, 4, 2)
    position = NP.concatenate((NP.min(position, axis=1), NP.max(position, axis=1)), axis=1)
    
    return position
    

def loadSequence(path, extension, boxesFile):
    # Load frames
    framePaths = getFramePaths(path, extension)
    frame = loadFrames(framePaths)
    
    # Load bounding boxes information
    boxesPath = os.path.join(path, boxesFile)
    position = loadPosition(boxesPath)
    
    return frame, position
    
    
def loadSequences(path, names, extension, boxesFile, size):
    for name in names:
        seqPath = os.path.join(path, name)
        yield (name, ) + loadSequence(seqPath, extension, boxesFile, size)
    
    
def createDataset(path, seqs, extension, boxesFile, size, samples, seqLength, outPath):
    frames = NP.empty((samples, seqLength) + size + (3, ) )
    positions = NP.empty((samples, seqLength, 4))
    
    for i in range(samples):
        seqPath = os.path.join(path, random.choice(seqs))
        frame, position = loadSequence(seqPath, extension, boxesFile, size)
        start = NP.random.randint(0, frame.shape[0] - seqLength)
        frames[i, ...] = frame[start:start+seqLength, ...]
        positions[i, ...] = position[start:start+seqLength, ...]
        
    dataSetFile = open(outPath, "wb")
    dill.dump((frames, positions), dataSetFile)
    dataSetFile.close()
    

def loadDataset(path):
    dataSetFile = open(path, "rb")
    frame, position = dill.load(dataSetFile)
    dataSetFile.close()
    
    return frame, position
    
    
def generateVideo(pm, frames, gtP, predP, output, fps):
    gtPos = pm.fromTwoCorners(gtP).tolist()
    predPos = pm.fromTwoCorners(predP).tolist()
    images = Preprocess.getFrames(frames)
    sequence = Sequence(images, pm)
    sequence.addBoxes(gtPos, "Red")
    sequence.addBoxes(predPos, "Blue")
    sequence.exportToVideo(fps, output)
    

def plotResults(measures, measureName):
    cols = 2
    rows = len(measures) / cols + (len(measures) % cols > 0)
    f, axs = PLT.subplots(rows, cols, figsize=(20, 28))
    axs = axs.ravel()
    i = 0
    for name, values in measures.iteritems():
        ax = axs[i]
        ax.plot(values[0, ...])
        value = str(values[0, ...].mean())
        title = "{}, {} {} = {}".format(name, measureName, "Mean", value)
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel(measureName)
        print(title)
        i += 1
