class Config:
    def __init__(self):
        self.res=(100,100)
        #1 for greyscale, 3 for color
        self.imagChannels=3
        self.trainBatchSize=200
        self.testBatchSize=20
        self.epochs=10

        #percentage of training data used for testing
        self.testPercent=0.1
        self.trainingDataName="training-data.npy"
        self.trainingMetadataName="training-meta-data.npy"

        self.dataDir="raw-images"

        self.mirrorDataPoints=True

