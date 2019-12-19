class Hyperparameters:
    def __init__(self):
        #default values are from a sentdex tutorial, however they are fully configurable
        self.convLayers=3
        #first channel is cnn input and is decided by the image channels
        self.convChannels=[32,64,128]
        self.convKernals=[5,5,5]
        self.maxPoolWindows=[(2,2),(2,2),(1,1)]

        self.fcLayers=2
        #input into full con is decided by flatting last conv output
        #output of full con is decided by output of the network
        self.fcNodes=[512]
        
        #note learning rate should not be set too high because 
        #of dead relu problems this is also effected by higher batch sizes
        self.learningRate=0.001
        
        