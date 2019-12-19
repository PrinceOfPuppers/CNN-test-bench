import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from cnn import Cnn
from config import Config
from hyperparameters import Hyperparameters
import matplotlib.pyplot as plt




if __name__=="__main__":
    cfg=Config()
    data=np.load(cfg.trainingDataName,allow_pickle=True)
    metaData=np.load(cfg.trainingMetadataName,allow_pickle=True)
    numSubjects=len(metaData[0])
    print("Dataset Loaded")
    
    hyp=Hyperparameters()
    cnn=Cnn(cfg,hyp,numSubjects)
    print("CNN Generated")


    
    images=((torch.from_numpy(np.stack(data[:,0]))).view(-1,cfg.imagChannels,cfg.res[0],cfg.res[1]))
    oneHotVecs=torch.Tensor(list(data[:,1]))
    print("Images Converted to Tensors")

    
    trainSize=int((1-cfg.testPercent)*len(data))
    testSize=len(data)-trainSize

    trainImgs=images[:trainSize]
    trainOneHotVecs=oneHotVecs[:trainSize]

    testImgs=images[trainSize:]
    testOneHotVecs=oneHotVecs[trainSize:]


    trainingAccuracy=[]
    trainingLoss=[]

    testingAccuracy=[]
    testingLoss=[]

    epochs=list(range(0,cfg.epochs))
    #training
    for epoch in range(0,cfg.epochs):

        #records weights to see if they change after training
        weights1=[]
        for param in cnn.parameters():
            weights1.append(param.clone())

        #training
        print("Training Epoch",str(epoch+1),"of",str(cfg.epochs)+":")
        trainAcc,trainLoss=cnn.train(trainImgs,trainOneHotVecs,cfg.trainBatchSize)
        trainingAccuracy.append(trainAcc)
        trainingLoss.append(trainLoss)

        #prints if weights are unchanged
        weights2=[]
        unchangedWeights=0
        for param in cnn.parameters():
            weights2.append(param.clone())
        for i in zip(weights1, weights2):
            if torch.equal(i[0], i[1]):
                unchangedWeights+=1
                
        if unchangedWeights>0:
            print(unchangedWeights,"of",len(weights1),"Weights Unchanged")


        print("Testing")
        testAcc,testLoss=cnn.test(testImgs,testOneHotVecs,cfg.testBatchSize)

        testingAccuracy.append(testAcc)
        testingLoss.append(testLoss)
    
    #plots accuracy
    plt.plot(epochs,trainingAccuracy, label="Training Accuracy",color="green")
    plt.plot(epochs,testingAccuracy, label="Testing Accuracy",color="blue")
    plt.plot(epochs,[1/numSubjects]*cfg.epochs,label="Blind Guessing Accuracy", color="purple")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy (Decimal)")
    plt.title("Accuracy Graph\n"+str(cfg.epochs)+" Epochs, "+str(numSubjects)+ " Subjects, "+str(trainSize)+" Training Data Points")
    plt.legend()
    plt.show()

    #plots loss
    plt.plot(epochs,trainingLoss, label="Training Loss",color="orange")
    plt.plot(epochs,testingLoss, label="Testing Loss",color="red")
    plt.xlabel("epochs")
    plt.title("Loss Graph\n"+str(cfg.epochs)+" Epochs, "+str(numSubjects)+ " Subjects, "+str(trainSize)+" Training Data Points")
    plt.legend()
    plt.show()