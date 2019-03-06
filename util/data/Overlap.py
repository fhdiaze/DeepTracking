# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:46:25 2016

@author: MindLab
"""

import numpy as NP
import theano.tensor as THT
from keras import backend as K
from tracking.model.core.Measure import Measure

class Overlap(Measure):
    
    def __init__(self):
        super(Overlap, self).__init__("Overlap")
    
    # gtPosition.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def calculate(self, gtPosition, predPosition):
        left = NP.max([predPosition[..., 0], gtPosition[..., 0]], axis=0)
        top = NP.max([predPosition[..., 1], gtPosition[..., 1]], axis=0)
        right = NP.min([predPosition[..., 2], gtPosition[..., 2]], axis=0)
        bottom = NP.min([predPosition[..., 3], gtPosition[..., 3]], axis=0)
        intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
        label_area = NP.abs(gtPosition[..., 2] - gtPosition[..., 0]) * NP.abs(gtPosition[..., 3] - gtPosition[..., 1])
        predict_area = NP.abs(predPosition[..., 2] - predPosition[..., 0]) * NP.abs(predPosition[..., 3] - predPosition[..., 1])
        union = label_area + predict_area - intersect
        iou = intersect / union
                
        return iou
        
        
    # gtPosition.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def calculateGpu(self, gtPosition, predPosition):
        pShape = K.shape(gtPosition)
        inputDim = K.ndim(gtPosition)
        gtPosition = K.reshape(gtPosition, (-1, pShape[-1]))
        predPosition = K.reshape(predPosition, (-1, pShape[-1]))
        left = K.maximum(predPosition[:, 0], gtPosition[:, 0])
        top = K.maximum(predPosition[:, 1], gtPosition[:, 1])
        right = K.minimum(predPosition[:, 2], gtPosition[:, 2])
        bottom = K.minimum(predPosition[:, 3], gtPosition[:, 3])
        intersect = (right - left) * ((right - left) > 0) * (bottom - top) * ((bottom - top) > 0)
        label_area = K.abs(gtPosition[:, 2] - gtPosition[:, 0]) * K.abs(gtPosition[:, 3] - gtPosition[:, 1])
        predict_area = K.abs(predPosition[:, 2] - predPosition[:, 0]) * K.abs(predPosition[:, 3] - predPosition[:, 1])
        union = label_area + predict_area - intersect
        iou = intersect / union
        #iouShape = K.concatenate([pShape[:-1], (1, )])
        iou = THT.reshape(iou, (pShape[0], pShape[1], 1), ndim=inputDim)
                
        return iou