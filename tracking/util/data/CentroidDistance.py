# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 19:50:54 2016

@author: MindLab
"""

import numpy as NP
from tracking.model.core.Measure import Measure
from tracking.util.model.CentroidPM import CentroidPM

class CentroidDistance(Measure):
    
    
    def __init__(self):
        super(CentroidDistance, self).__init__("CentroidDistance")
        self.centroid = CentroidPM(15, 10)
    
    
    def calculate(self, gtPosition, predPosition):
        gtPosition = self.centroid.fromTwoCorners(gtPosition)
        predPosition = self.centroid.fromTwoCorners(predPosition)
        distance = NP.linalg.norm(gtPosition - predPosition, axis=2)
        
        return distance