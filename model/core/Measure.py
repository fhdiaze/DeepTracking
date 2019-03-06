# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 18:42:31 2016

@author: MindLab
"""

class Measure(object):
    
    """
    Create a measure calculator.

    @type  name: String
    @param name: The name of the measure
    """
    def __init__(self, name):
        self.name = name
    
    
    """
    Calculate a measure value.

    @type  gtPosition: numpy.array(batchSize, seqLength, targetDim(x1, y1, x2, y2))
    @param gtPosition: The ground truth position in two corners representation
    @type  predPosition: numpy.array(batchSize, seqLength, targetDim(x1, y1, x2, y2))
    @param predPosition: The predicted position in two corners representation
    @rtype:  (float, numpy.array(batchSize, seqLength, value))
    @return: The measure value
    """
    def calculate(self, gtPosition, predPosition):
        pass