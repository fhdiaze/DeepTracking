# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:18:20 2016

@author: MindLab
"""

import matplotlib.pyplot as PLT

def getValues(logPath):
    epochMeasures = {}
    batchMeasures = []
    batchLosses = []
    
    with open(logPath, mode="r") as log:
        for line in log:
            if "Validation Epoch -" in line:
                name = line.split("-")[-1].split(":")[0].strip()
                value = float(line.split("=")[-1])
                
                if epochMeasures.get(name, None) is None:
                    epochMeasures[name] = []

                epochMeasures[name].append(value) 
                
            if "Validation Batch:" in line:
                value = float(line.split("=")[-1])
                batchMeasures.append(value)
        
            if "Batch Loss:" in line:
                loss = line.split(":")[-1].split(",")[-1]
                loss = float(loss.split("=")[-1])
                batchLosses.append(loss)
                
    return batchLosses, batchMeasures, epochMeasures
    
    
def plotEpochMeasures(logPath, figPath):
    batchLosses, batchMeasures, epochMeasures = getValues(logPath)
    fig = PLT.figure(1, (20., 10.))
    PLT.plot(epochMeasures, color="g")
    PLT.title(logPath)
    fig.savefig(figPath)
    PLT.close(fig)