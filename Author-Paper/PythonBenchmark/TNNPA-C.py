'''
Created on Nov 7, 2012

@author: bhanu
'''
import sys
from array import array
import numpy as np
import os
import math


class Layer(object):
    '''
    Represents one single layer in a Mlp
    '''
    
    def __init__(self, nNeurons, nInpsPerNeuron, transferF, ilayer):
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
        if(ilayer != 0):    #if this is not an input layer
            self.W = (4)*np.random.random_sample(size=(nInpsPerNeuron+1,nNeurons)) - 2     #W[0,i] beingh the BIAS weights 
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
        self.learningRate = learningRate
        
    def trainMlp(self, dataFile, learningRate=0.1, epochs=1000):
        '''
        Trains this Mlp with the training data
        '''
        
        trainSet = getTrainingData(dataFile)
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
            for layer in mlp.layers[1:] :           #for all layers starting from the second layer
                for i in range(layer.nNeurons):
                    layer.net[i] =  weightedSum(outp, layer.W[1:,i]) + layer.W[0,i]
                    layer.out[i] = g(layer.net[i], layer.transferF)   #pass this weighted sum through the transfer function of this layer                  
                outp = layer.out
            print "output = ", mlp.layers[-1].out
        
    
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
            
            
def getTrainingData(dataFile):
    #----------prepare training data from the dataFile---------
    head, tail = os.path.split(dataFile)
    if(head == ''):
        cwd = os.path.curdir
        trainingFile = os.path.join(cwd,tail)
    f = open(trainingFile)
    
    trainSet = []  #training samples
    lines = f.readlines()
    if(len(lines) > 1000):
        terminate("File Contains more than 1000 samples")
        
    for line in lines:
        if(line[0] == '#'):
            continue
        X = []
        Y = []  #list of inputs and oupts
        x_y = line.split('    ')     #Split the string in X(inputs) and Y(outputs), separated by tab
        x = x_y[0].strip()
        y = x_y[1].strip()
        xstr = x.split()     #split inputs with space
        ystr = y.split()     #split outputs with space
        for inp in xstr:
            X.append(float(inp))
        for outp in ystr:
            Y.append(float(outp))
        trainSet.append((X,Y))
        #print trainSet
    return trainSet
    

def terminate(msg):
    print """
    
    Please run the program with valid arguments. 
    
    USAGE:    $ python TNN_PA_A N M dataFile
                
                where,
                N         :    Dimension of Input Layer (x), less than 101
                M         :    Dimension of Output Layer (y), less than 30
                InputFile :    Name of the file containing training data, if 
                               not in current working directory of program then
                               provide fully qualified path, Maximum 200 samples
    Example:    $ python TNNPA_B 4 2 training.dat
    
    """
    sys.exit(msg)

    
def trainBPE(mlp, trainSet, learningRate, maxEpoch):
    '''
    Training of Multi-layer perceptron using Backpropagation of Error
    
    Parameters:-
    mlp:    Object of Mlp class
    trainSet:    List of training tuples, 
                 use method 'getTrainingData()' to get a valid training 
                 set from a training data file
    
    '''
    iteration = 1
    f = open('learning.curve', 'w')
    f.write('#Epoch-Number       #Mean Maximum Single Error \n')
    while(True):
        meanMaxError = maxerror = 0
        for x, y in trainSet : 
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
        
        meanMaxError = maxerror/len(trainSet)   
        f.write(str(iteration)+'        '+str(meanMaxError)+'\n')     
        if(iteration > maxEpoch):
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

if __name__ == '__main__':
    
    N = 4           #N : number of inputs/neurons for input layer
    H1 = 10          #H : number of neurons in hidden layer-1
    #H2 = 5
    M = 2           #number of outputs/neurons of the output layer
    dataFile = 'training.dat'
    learningRate = 0.1
    epochs =  5000
    
    #define layers of MLP keeping in mind that output of one layer is the number of inputs for the next layer
    layer0 = Layer(nNeurons=N, nInpsPerNeuron=-1, transferF='identity', ilayer=0)           #input layer
    layer1 = Layer(nNeurons=H1, nInpsPerNeuron=N, transferF='tanh', ilayer=1)                #hidden layer 1
    layer2 = Layer(nNeurons=M, nInpsPerNeuron=H1, transferF='tanh', ilayer=2)                #output layer 
    #layer3 = Layer(nNeurons=M, nInpsPerNeuron=H2, transferF='logistic', ilayer=3)            #output layer
    
    layers = [layer0, layer1, layer2 ]
    
    mlp = Mlp(layers)
    mlp.showMlp()
    print "\n\nTraining  Mlp for", epochs," Epochs.... please wait... "   
    trainedMlp, iterations = mlp.trainMlp(dataFile, learningRate, epochs)
    print "\n\nFinished training of Mlp "
    trainedMlp.showMlp()
    mlp.test()