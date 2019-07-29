# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 22:07:36 2017

@author: kuby
"""

def loadPosition(path):
    # Load bounding boxes information
    position = []
    
    with open(path) as f:
        [position.append(list(map(float, line.split(" ")))) for line in f]
    
    position = NP.array(position)
    anns = position[:, 0]
    position = position[:, 1:]
    
    return anns.astype(int), position

seqs = [o for o in os.listdir(seqsPath) if os.path.isdir(os.path.join(seqsPath,o))]
anns = [o for o in os.listdir(seqsPath) if os.path.isfile(os.path.join(seqsPath,o))]

for seq in seqs:
    folder = os.path.join(seqsPath, seq)
    
    framePaths = VotTool.getFramePaths(folder, ".jpg")
    path = os.path.join(folder, "boxes.txt")
    anns, bbox = loadPosition(path)
    
    f = interp1d(anns, bbox, axis=0)
    newx = NP.arange(anns[0], anns[-1]+1)
    newBbox = f(newx)
    
    fullBbox = NP.zeros((len(framePaths), 8))
    fullBbox[newx-1, ...] = newBbox
    
    bboxPath = os.path.join(folder, "groundtruth.txt")
    annsPath = os.path.join(folder, "newAnns.txt")
    
    NP.savetxt(bboxPath, fullBbox, fmt='%1.8f', delimiter=",")
    NP.savetxt(annsPath, newx, fmt='%i')
    