import os
import cv2
import numpy as np
from tqdm import tqdm
from random import random
import matplotlib.pyplot as plt
from config import Config
def loadImgToArray(path,res,imChannels):
    #path is rel path, res is tuple
    if imChannels==1:
        image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    elif imChannels==3:
        image=cv2.imread(path,cv2.IMREAD_COLOR)
    else:
        raise Exception("Invalid Input Channel Number "+str(imChannels))
                
    image=cv2.resize(image,res)
    image=np.array(image/255,dtype=np.float32)
    return(image)

class DataProcessor:
    def __init__(self,cfg):
        self.res=cfg.res
        self.imChannels=cfg.imagChannels

        self.dataDir=cfg.dataDir
        self.subjects=[]
        self.relPaths=[]

        #gets the relative paths and subject labels
        self.getSubjects()

        self.balanceChecker=[0]*len(self.subjects)
        self.failedLoadCount=[0]*len(self.subjects)

        self.mirrorDataPoints=cfg.mirrorDataPoints
        self.trainingData=[]
        self.trainingDataName=cfg.trainingDataName
        self.trainingMetadataName=cfg.trainingMetadataName

        

        
    def getSubjects(self):
        subjectCounter=0
        for fileName in os.listdir(self.dataDir):
            subjectCounter+=1
            self.subjects.append(fileName)
            self.relPaths.append(os.path.join(self.dataDir,fileName))
        if subjectCounter<=1:
            print("Not Enough subjects in",os.path.join(os.getcwd(),self.dataDir))
    

    def generateTrainingData(self):
        #subject loop
        for subjectNum in range(0,len(self.subjects)):

            subject=self.subjects[subjectNum]
            #subject vec is one hot vector ie) [0,0,0,1,0,0,0,0]
            subjectVec=np.eye(len(self.subjects))[subjectNum]

            print("Loading Images of",subject)
            print("Number",subjectNum+1,"of",str(len(self.subjects))+":")

            #individual image loop
            for file in tqdm(os.listdir(self.relPaths[subjectNum])):
                #try catch block for bad image data
                try:
                    path=self.relPaths[subjectNum]+"/"+file

                    image=loadImgToArray(path,self.res,self.imChannels)

                    self.trainingData.append([image,subjectVec])

                    if self.mirrorDataPoints:
                        self.trainingData.append([np.flip(image,axis=1),subjectVec])
                        self.balanceChecker[subjectNum]+=2
                    else:
                        self.balanceChecker[subjectNum]+=1

                except:
                    if self.mirrorDataPoints:
                        self.failedLoadCount[subjectNum]+=2
                    else:    
                        self.failedLoadCount[subjectNum]+=1

        #Result Printout
        print("Training Data Generated With:")
        for subjectNum in range(0,len(self.subjects)):
            print("-",self.balanceChecker[subjectNum], self.subjects[subjectNum],"and",self.failedLoadCount[subjectNum],"failed Loads")

        #Shuffle Data
        print("Shuffling TrainingData")
        np.random.shuffle(self.trainingData)

        #Save Data and Metadata
        print("Saving TrainingData as",self.trainingDataName)
        np.save(self.trainingDataName,self.trainingData)
        print("Saving Metadata as",self.trainingMetadataName)
        np.save(self.trainingMetadataName,[self.subjects,self.balanceChecker])

        print("Training Data Generating Complete with",len(self.trainingData),"data points")

    def printTrainingDataStats(self):
        try:
            self.trainingData=np.load(self.trainingDataName,allow_pickle=True)
            self.trainingMetadata=np.load(self.trainingMetadataName,allow_pickle=True)
            subjects=self.trainingMetadata[0]
            balance=self.trainingMetadata[1]
            print("training data with",len(self.trainingData),"data points")
            print("subjects and balance:")
            for i,subject in enumerate(subjects):
                print("-",subject,balance[i])

            exampleNum=int(len(self.trainingData)*random())
            print("Example Datapoint number",exampleNum,"Subject Type,",self.subjects[np.argmax(self.trainingData[exampleNum][1])])
            if self.imChannels==1:
                plt.imshow(self.trainingData[exampleNum][0],cmap="gray")
            else:
                plt.imshow(self.trainingData[exampleNum][0])
            plt.show()
        except:
            print("No Training Data Generated")

        

if __name__=="__main__":
    cfg=Config()
    app=DataProcessor(cfg)
    
    genNewData=str.lower(input("Would you like to regenerate the Data? (y/n): "))
    if genNewData=="y":
        app.generateTrainingData()
    app.printTrainingDataStats()
