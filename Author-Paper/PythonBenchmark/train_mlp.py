import data_io
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import math
import os
import numpy as np
import cPickle


class Layer(object):
    '''
    Represents one single layer in a Mlp
    '''
    
    def __init__(self, nNeurons, nInpsPerNeuron, transferF, ilayer, seed=0):
        '''
        Each layer has:
        nNeurons:            Number of neurons
        nInpsPerNeuron:      Number of inputs per Neuron, not needed for input layer so use '-1' for input layer
        transferF:           Transfer Function, which could be 'tanh', 'logistic' or 'identity'
        ilayer:              Index of the layer
        '''
        self.nNeurons  = nNeurons
        self.transferF = transferF
        self.ilayer = ilayer
        self.nInpsPerNeuron = nInpsPerNeuron
        np.random.seed(seed)
        if(ilayer != 0):    #if this is not an input layer
            self.W = (1)*np.random.random_sample(size=(nInpsPerNeuron+1,nNeurons)) - 0.5    #W[0,i] being the BIAS weights 
            self.W[0,:] = -0.5  #Bias Weight
            self.net = np.zeros(nNeurons)    #place holder vector for Net i.e. weighted sum for each neuron of this layer
            self.out = np.zeros(nNeurons)     #place holder vector for Output of each neuron of this layer
            self.delta = np.zeros(nNeurons)   #place holder vector for delta of this layer


class Mlp(object):
    '''
    Represents a Multi Layer Perceptron Network
    '''

    def __init__(self, layers):
        '''
        Constructor
    
        Parameters:
                    Layers:    List of 'Layer' objects
        '''
        self.layers = layers
        #self.learningRate = learningRate
        
    def trainMlp(self, trainSet, learningRate=0.1, epochs=1000):
        '''
        Trains this Mlp with the training data
        '''
        
        #trainSet = getTrainingData(dataFile)
        return trainBPE(self,trainSet, learningRate, epochs) 

    
    def test(self):
        '''
        Test the trained Mlp network
        '''
        while(True):
            print"\n\nTesting trained perzeptron network, press Ctrl+c to quit"
            Xraw =  raw_input("Enter inputs separated by space to test this trained Mlp: ")
            Xlist = Xraw.strip().strip('\n').split(' ')
            X = [float(x) for x in Xlist]
            #Propagate the inputs forward to compute the outputs             
            outp = list(X)     #output of  input layer i.e. output of previous layer to be used as input for next layer
            for layer in self.layers[1:] :           #for all layers starting from the second layer
                for i in range(layer.nNeurons):
                    layer.net[i] =  weightedSum(outp, layer.W[1:,i]) + layer.W[0,i]
                    layer.out[i] = g(layer.net[i], layer.transferF)   #pass this weighted sum through the transfer function of this layer                  
                outp = layer.out
            print "output = ", self.layers[-1].out
        
    
    def showMlp(self):
        '''
        Print all the layers of this perzeptron
        '''
        for layer in self.layers:
            print 'Layer ', layer.ilayer
            print 'Number of Neurons: ', layer.nNeurons
            print 'Transfer Function: ', layer.transferF
            if(layer.ilayer != 0): 
                print 'Weights(',layer.W.shape,'): ', layer.W
            print '\n'

def trainBPE(mlp, trainSet, learningRate, maxEpoch):
    '''
    Training of Multi-layer perceptron using Backpropagation of Error
    
    Parameters:-
    mlp:    Object of Mlp class
    trainSet:    List of training tuples, 
                 use method 'getTrainingData()' to get a valid training 
                 set from a training data file
    
    '''
    
#    train = [t for t in trainSet[indices[:-10]] ]
#    test = [t for t in trainSet[indices[-10:]] ]#trainSet[indices[-10:]]
#    
    
    iteration = 1
    f = open('learning.curve', 'w')
    f.write('#Epoch-Number       #Mean Maximum Single Error \n')
    np.random.seed(0)
    while(True):
        #split data into train and test sets for crossvalidation        
        indices = np.random.permutation(len(trainSet))
        train = []; test = []
        for i in indices[:-10]:
            train.append(trainSet[i])
        for i in indices[-10:]:
            test.append(trainSet[i])
            
        
        meanMaxError = 0; maxerror = 0; maxtesterror = 0; bestTestError = np.Inf; patience = 0
        for x, y in train : 
            #Propagate the inputs forward to compute the outputs             
            outp = list(x)     #output of  input layer i.e. output of previous layer to be used as input for next layer
            for layer in mlp.layers[1:] :           #for all layers starting from the second layer
                for i in range(layer.nNeurons):
                    layer.net[i] =  weightedSum(outp, layer.W[1:,i]) + layer.W[0,i]
                    layer.out[i] = g(layer.net[i], layer.transferF)   #pass this weighted sum through the transfer function of this layer                  
                outp = layer.out
                
            #Propagate deltas backward from output layer to input layer 
            layer = mlp.layers[-1]           
            for m in range(layer.nNeurons):        #for neurons in output layer 
                layer.delta[m] = derivativeOfG(layer.net[m], layer.transferF) * (y[m] - layer.out[m])   
            deltaP = layer.delta  # delta of a layer to be used by a layer above it, starting from output layer
            for l in range(len(mlp.layers)-2,0,-1) :  # for all hidden layers until input layer
                thislayer = mlp.layers[l]
                layerbelow = mlp.layers[l+1]
                for h in range(layer.nNeurons):
                    thislayer.delta[h] = derivativeOfG(thislayer.net[h], thislayer.transferF) * weightedSum(deltaP, layerbelow.W[h+1,:]) 
                deltaP = thislayer.delta  # for the next layer
                    
            #Update every weight in network using deltas 
            out_i = list(x)
            for layer in mlp.layers[1:] :
                #update current weights                     
                for i, inp in enumerate(out_i):
                    for j in range(layer.nNeurons):
                        layer.W[i+1,j] += learningRate * (inp * layer.delta[j]) 
                out_i = layer.out
      
            error = [math.fabs(value) for value in y - mlp.layers[-1].out ] 
            maxerror += max(error)           
        
        meanMaxError = maxerror/len(train)
        print "Train set MeanError:  ", meanMaxError   
        f.write(str(iteration)+'        '+str(meanMaxError)+'\n')   
        #calculate mean test error:
        for x, y in test : 
            #Propagate the inputs forward to compute the outputs             
            outp = list(x)     #output of  input layer i.e. output of previous layer to be used as input for next layer
            for layer in mlp.layers[1:] :           #for all layers starting from the second layer
                for i in range(layer.nNeurons):
                    layer.net[i] =  weightedSum(outp, layer.W[1:,i]) + layer.W[0,i]
                    layer.out[i] = g(layer.net[i], layer.transferF)   #pass this weighted sum through the transfer function of this layer                  
                outp = layer.out  
            testerror = [math.fabs(value) for value in y - mlp.layers[-1].out ] 
            maxtesterror += max(error)   
        meanTestError = maxtesterror/(len(test))   
        print "Test set Mean Error: ", meanTestError
        if(meanTestError < bestTestError):
            bestTestError = meanTestError
            patience = 0
        else:
            patience +=1
                 
        if(iteration > maxEpoch or patience > 2 ):
            print "Exiting with patience: ", patience, " and iteration: ", iteration
            break
        iteration += 1
    
    f.close()
    return mlp, iteration



def g(inp, transferF):
    if transferF == 'tanh':
        value = math.tanh(inp)
    elif transferF == 'identity':
        value = inp
    elif transferF == 'logistic':
        value = 1 / (1 + math.exp(-inp))
    else :
        raise ValueError('Invalid transfer function type: ', transferF)
    
    return value

def isStoppingCriterion():
    return False

def derivativeOfG(inp, transferF):
    if transferF == 'tanh':
        temp = math.tanh(inp)
        value = 1 - temp*temp   # 1 - tanh^2
    elif transferF == 'identity':
        value = 0  # derivative of Identity function is zero
    elif transferF == 'logistic':
        temp = 1 / (1 + math.exp(-inp))
        value = temp*(1-temp)  # derivative of logistic function is f*(1-f)
    else :
        raise ValueError('Invalid transfer function type: ', transferF)
    
    return value


def weightedSum(inputVector, weights):
#        print inputVector
#        print weights
    sum = (np.sum(inputVector*weights)) 
#        print sum
    return sum


def main():
    print("Getting features for deleted papers from the database")
    if(os.path.exists("features_deleted.obj")):
        with open("features_deleted.obj", 'r') as loadfile:
            features_deleted = cPickle.load(loadfile)
    else:
        features_deleted = data_io.get_features_db("TrainDeleted")
        with open("features_deleted.obj", 'w') as dumpfile:
            cPickle.dump(features_deleted, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)

    print("Getting features for confirmed papers from the database")
    if(os.path.exists("features_confirmed.obj")):
        with open("features_confirmed.obj", 'r') as loadfile:
            features_conf = cPickle.load(loadfile)
    else:
        features_conf = data_io.get_features_db("TrainConfirmed")
        with open("features_confirmed.obj", 'w') as dumpfile:
            cPickle.dump(features_conf, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)

    features = [x[2:] for x in features_deleted + features_conf]
    target = [[0] for x in range(len(features_deleted))] + [[1] for x in range(len(features_conf))]
    
    featuresInts = []
    for tup in features:
        a, b, c, d, e = tup
        featuresInts.append((int(a), int(b), int(c), int(d), int(e)))

   
    trainSet = zip(featuresInts, target)
    

   
    N = 5          #N : number of inputs/neurons for input layer
    H1 = 100       #H : number of neurons in hidden layer-1
    #H2 = 5
    M = 1           #number of outputs/neurons of the output layer
    
    learningRate = 0.1
    epochs =  1000
    
    #define layers of MLP keeping in mind that output of one layer is the number of inputs for the next layer
    layer0 = Layer(nNeurons=N, nInpsPerNeuron=-1, transferF='identity', ilayer=0, seed=13)           #input layer
    layer1 = Layer(nNeurons=H1, nInpsPerNeuron=N, transferF='tanh', ilayer=1, seed=13)                #hidden layer 1
    layer2 = Layer(nNeurons=M, nInpsPerNeuron=H1, transferF='tanh', ilayer=2, seed=13)                #output layer 
    #layer3 = Layer(nNeurons=M, nInpsPerNeuron=H2, transferF='logistic', ilayer=3)            #output layer
    
    layers = [layer0, layer1, layer2 ]
    
    mlp = Mlp(layers)
    mlp.showMlp()
    print "\n\nTraining  Mlp for", epochs," Epochs.... please wait... "   
    trainedMlp, iterations = mlp.trainMlp(trainSet, learningRate, epochs)
    print "\n\nFinished training of Mlp "
    trainedMlp.showMlp()
    
    print("Saving the classifier")
    data_io.save_model(mlp,prefix="mlp_")
    
    
if __name__=="__main__":
    main()