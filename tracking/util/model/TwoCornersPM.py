# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:37:32 2016

@author: MindLab
"""

import numpy as NP
from PIL import ImageDraw
from tracking.model.core.PositionModel import PositionModel

class TwoCornersPM(PositionModel):
    
    def __init__(self):
        super(TwoCornersPM, self).__init__(4)
    
    
    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def fromTwoCorners(self, position):
        
        return position
    
    
    # position.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def toTwoCorners(self, position):
        
        return position
    
        
    """
    Plot (in side) the position in a frame. 

    @type  frame:    PIL.Image
    @param frame:    The frame
    @type  position: [integer, integer, integer, integer]
    @param position: The objects's corners coordinates [x1, y1, x2, y2]
    @type  outline:  string
    @param outline:  The name of the position color.
    """ 
    def plot(self, frame, position, outline):
        draw = ImageDraw.Draw(frame)
        
        draw.rectangle(position, outline=outline)
        
        
    # gtPosition.shape = (batchSize, seqLength, targetDim(x1, y1, x2, y2))
    def calculateIOU(self, gtPosition, predPosition):
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