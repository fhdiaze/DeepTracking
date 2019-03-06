# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:30:20 2016

@author: fhdiaze
"""

class SimulatorFactory():

    def __init__(self, dataDir, trajectoryModelSpec, cameraTrajectoryModelSpec, summaryPath, gmmPath=None, scenePathTemplate = 'images/train2014', objectPathTemplate = 'images/train2014'):
        self.dataDir = dataDir
        self.scenePathTemplate = scenePathTemplate
        self.objectPathTemplate = objectPathTemplate
        print 'Loading summary from file {}'.format(summaryPath)
        summaryFile = open(summaryPath, 'r')
        self.summary = pickle.load(summaryFile)
        summaryFile.close()
        #Default model is Random, overwrite if specified
        self.trajectoryModelSpec = trajectoryModelSpec
        self.cameraTrajectoryModelSpec = cameraTrajectoryModelSpec
        self.gmmModel = None
        if gmmPath is not None:
            gmmFile = open(gmmPath, 'r')
            self.gmmModel = pickle.load(gmmFile)
            gmmFile.close()
        self.sceneList = os.listdir(os.path.join(self.dataDir, self.scenePathTemplate))

    def createInstance(self, *args, **kwargs):
        '''Generates TrajectorySimulator instances with a random scene from the scene template and a random object from the object template'''
        self.randGen = startRandGen()
        #Select a random image for the scene
        scenePath = os.path.join(self.dataDir, self.scenePathTemplate, self.randGen.choice(self.sceneList))

        #Select a random image for the object
        objData = self.randGen.choice(self.summary[SUMMARY_KEY])
        objPath = os.path.join(self.dataDir, self.objectPathTemplate, objData['file_name'].strip())

        #Select a random object in the scene and read the segmentation polygon
        logging.debug('Segmenting object from category %s', self.summary[CATEGORY_KEY][int(objData['category_id'])])
        polygon = self.randGen.choice(objData['segmentation'])

        simulator = TrajectorySimulator(scenePath, objPath, polygon=polygon, trajectoryModelSpec=self.trajectoryModelSpec, cameraTrajectoryModelSpec=self.cameraTrajectoryModelSpec, gmmModel=self.gmmModel, *args, **kwargs)
        
        return simulator