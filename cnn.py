import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
from tqdm import tqdm

class Cnn(nn.Module):
    def __init__(self,cfg,hyp,numSubjects):
        super().__init__()
        self.inputRes=cfg.res
        self.imagChannels=cfg.imagChannels
        self.maxPoolWindows=hyp.maxPoolWindows

        #convolutional layers
        self.convLayers=nn.ModuleList()

        convLayer=nn.Conv2d(self.imagChannels,hyp.convChannels[0],hyp.convKernals[0])
        self.convLayers.append(convLayer)

        for i in range(0,hyp.convLayers-1):
            convLayer=nn.Conv2d(hyp.convChannels[i],hyp.convChannels[i+1],hyp.convKernals[i+1])
            self.convLayers.append(convLayer)
        
        #determines the number of input nodes for the first fully
        #connected layer
        self.flattenDim=self.determineFlatten()

        #fully connected (dense) layers
        self.fcLayers=nn.ModuleList()
        for i in range(0,hyp.fcLayers):
            if i==0:
                inputNodes=self.flattenDim
            else:
                inputNodes=hyp.fcNodes[i-1]
            
            if i==hyp.fcLayers-1:
                outputNodes=numSubjects
            else:
                outputNodes=hyp.fcNodes[i]
            fcLayer=nn.Linear(inputNodes,outputNodes)
            self.fcLayers.append(fcLayer)


        #selects device depending on if cuda is avalible
        if torch.cuda.is_available():
            self.device=torch.device("cuda:0")
        else:
            self.device=torch.device("cpu")
        self.to(self.device)

        
        self.optimizer=optim.Adam(self.parameters(),hyp.learningRate)
        self.lossFunct=nn.MSELoss()

    
    def determineFlatten(self):
        x=torch.randn(1,self.imagChannels,self.inputRes[0],self.inputRes[1]).view(-1,self.imagChannels,self.inputRes[0],self.inputRes[1])
        for i,conv in enumerate(self.convLayers):
            x=funct.max_pool2d(funct.relu(conv(x)),self.maxPoolWindows[i])

        return x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

    def forward(self,x):
        for i,conv in enumerate(self.convLayers):
            x=funct.max_pool2d(funct.relu(conv(x)),self.maxPoolWindows[i])

        x=x.view(-1,self.flattenDim)

        for i,fcLayer in enumerate(self.fcLayers):
            #last fc layer shouldnt have relu called on it
            if i==len(self.fcLayers)-1:
                x=fcLayer(x)
            else:
                x=funct.relu(fcLayer(x))
        
        return funct.softmax(x, dim=1)

    
    def train(self,trainImgs,trainOneHotVecs,batchSize):
        trainSize=len(trainImgs)
        right=0
        avgLoss=0
        for i in tqdm(range(0,trainSize,batchSize)):
            imgBatch=trainImgs[i:i+batchSize].to(self.device)
            OneHotVecBatch=trainOneHotVecs[i:i+batchSize].to(self.device)
            
            self.zero_grad()
            outputs=self(imgBatch)
            loss=self.lossFunct(outputs,OneHotVecBatch)
            loss.backward()
            self.optimizer.step()

            avgLoss+=loss.item()*len(imgBatch)

            #calculates number of right answers
            for j,answerOneHot in enumerate(outputs):
                answer=torch.argmax(answerOneHot)
                correctAnswer=torch.argmax(OneHotVecBatch[j])
                if answer==correctAnswer:
                    right+=1 

        avgLoss=avgLoss/trainSize
        accuracy=right/trainSize
        print("Training Accuracy (decimal):",accuracy,"Training loss:",avgLoss)
        return (accuracy,avgLoss)
    
    def test(self,testImgs,testOneHotVecs,batchSize):
        right=0
        avgLoss=0
        testSize=len(testImgs)
        with torch.no_grad():
            for i in tqdm(range(0,testSize,batchSize)):
                imgBatch=testImgs[i:i+batchSize].to(self.device)
                OneHotVecBatch=testOneHotVecs[i:i+batchSize].to(self.device)
                outputs=self(imgBatch)
                loss=self.lossFunct(outputs,OneHotVecBatch)

                avgLoss+=loss.item()*len(imgBatch)

                #calculates number of right answers
                for j,answerOneHot in enumerate(outputs):
                    answer=torch.argmax(answerOneHot)
                    correctAnswer=torch.argmax(OneHotVecBatch[j])
                    if answer==correctAnswer:
                        right+=1 

        accuracy=right/testSize
        avgLoss=avgLoss/testSize
        print("Testing Accuracy (decimal):",accuracy,"Testing Loss:",avgLoss)
        return (accuracy,avgLoss)