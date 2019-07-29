# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:43:02 2016

@author: MindLAB
"""

import cPickle as pickle
import os as OS
import Tkinter as TK
from PIL import Image, ImageDraw, ImageTk

class ObjectFilter(object):
    
    def __init__(self, basePath, objSetPath, filtObjSetPath):
        self.basePath = basePath
        self.objSetPath = objSetPath
        self.filtObjSetPath = filtObjSetPath
        self.key = "summary"
        self.markKey = "mark"
        self.indx = 0
        self.objectSet = self.loadObjectSet(self.objSetPath)
        self.createFrame()
        self.loadNextSample()
        self.root.mainloop()
        
        
    def createFrame(self):
        self.root = TK.Tk()
        self.frame = TK.Frame(self.root)
        self.frame.pack()

        self.goodButton = TK.Button(
            self.frame, text="Good", fg="green", command=self.markAsGood
            )
        self.goodButton.pack(side=TK.LEFT)

        self.badButton = TK.Button(self.frame, text="Bad", fg="red", command=self.markAsBad)
        self.badButton.pack(side=TK.RIGHT)
        
        self.image = TK.Label(self.frame)
        self.image.pack(side=TK.TOP, fill="both", expand = "yes")
        
        # create a menu
        menu = TK.Menu(self.root)
        self.root.config(menu=menu)
        
        filemenu = TK.Menu(menu)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Save", command=self.save)
        filemenu.add_command(label="Save Filtered", command=self.saveFiltered)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.exitApp)
        

    def markAsGood(self):
        self.activeSample[self.markKey] = True
        self.loadNextSample()
        
        
    def markAsBad(self):
        self.activeSample[self.markKey] = False
        self.loadNextSample()
        
    
    def save(self):
        self.saveObjectSet(self.objSetPath)
        

    def saveFiltered(self):
        self.saveFilteredObjectSet(self.filtObjSetPath)
        
        
    def saveObjectSet(self, path):
        with open(path, 'w') as objectSetFile:
            pickle.dump(self.objectSet, objectSetFile)
            
            
    def saveFilteredObjectSet(self, path):
        filteredObjSet = self.objectSet.copy()
        objs = [obj for obj in filteredObjSet[self.key] if obj.get(self.markKey, None)]
        filteredObjSet[self.key] = objs
        
        print "Total good objects: " + str(len(objs))
        
        with open(path, 'w') as objectSetFile:
            pickle.dump(filteredObjSet, objectSetFile)
                    
        
    def exitApp(self):
        self.frame.quit()
        self.root.destroy()
        
        
    def loadObjectSet(self, path):
        
        with open(path, 'r') as objectSetFile:
            objectSet = pickle.load(objectSetFile)
            
        return objectSet
        
        
    def loadNextSample(self):
        self.activeSample = self.getNextSample()
        imagePath = OS.path.join(self.basePath, self.activeSample['file_name'].strip())
        polygon = self.activeSample["segmentation"][0]
        image = Image.open(imagePath)
        draw = ImageDraw.Draw(image)
        draw.polygon(polygon, outline="blue")

        
        img = ImageTk.PhotoImage(image)
        self.image.configure(image = img)
        self.image.image = img # keep a reference!
        self.image.pack()
        
        
    def getNextSample(self):
        
        while self.objectSet[self.key][self.indx].get(self.markKey, None) is not None and self.indx < len(self.objectSet[self.key]):
            self.indx += 1
            
        print self.indx
            
        obj = self.objectSet[self.key][self.indx]
        
        return obj
        
objectSetPath = "/home/fhdiaze/Data/Tracking/cocoTrainSummaryCategAndSideGt100SmplsAllCorrectedFiltered.pkl"
filtObjSetPath = "/home/fhdiaze/Data/Tracking/cocoTrainSummaryFilteredByHand.pkl"
basePath = "/home/datasets/MSCOCO/train2014/"
objFilt = ObjectFilter(basePath, objectSetPath, filtObjSetPath)