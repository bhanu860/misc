'''
Created on May 3, 2013

@author: bhanu
'''
from FeatureExtractor import FeatureExtractor
import theano.tensor as T
import os
import numpy
from theano.tensor.shared_randomstreams import RandomStreams
from AutoEncoder import AutoEncoder
import theano
import time
import cPickle
import scipy.sparse as sp




def test_AutoEncoder(learning_rate=0.1, training_epochs=15,
            batch_size=20):

    """

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

  
    """


    x_scipySparse = None; train_set_x = None; numInstances = 0; numFeatures = 0;
    if((os.path.exists("input_scipySparse.obj"))):
        print "loading sparse data from pickled file..."
        f = open("input_scipySparse.obj", 'r')
        x_scipySparse = cPickle.load(f)
        f.close()
        numInstances, numFeatures = x_scipySparse.shape
        
    else: 
        print "extracting features and building sparse data..."
        fe = FeatureExtractor()  
        fe.extractFeatures()
        train_set_x = fe.instanceList
        featureDict = fe.featDict   
        numInstances = len(train_set_x)
        numFeatures = len(featureDict)        
        x_lil = sp.lil_matrix((numInstances,numFeatures), dtype='float32') # the data is presented as a sparse matrix 
        i = -1; v = -1;
        try:
            for i,instance in enumerate(train_set_x):
                for v in instance.input:
                    x_lil[i, v] = 1
        except:
            print "i=",i," v=",v
        x_scipySparse = x_lil.tocsc()
        f = open("input_scipySparse.obj", 'w')
        cPickle.dump(x_scipySparse, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    

    # compute number of mini-batches for training, validation and testing
    n_train_batches = numInstances / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    #x = sparse.basic.as_sparse_variable(x_scipySparse, 'x')
    x = theano.shared(x_scipySparse, borrow=True)

    
    ####################################
    # BUILDING THE MODEL               #
    ####################################

    print "building the model..."
    rng = numpy.random.RandomState(123)

    ae = AutoEncoder(numpy_rng=rng, input=x, n_visible=numFeatures, n_hidden=10, n_trainExs=numInstances)

    cost, updates = ae.get_cost_updates(corruption_level=0.,
                                        learning_rate=learning_rate)

    train_ae = theano.function([index], cost, updates=updates,
         givens={x: train_set_x[index * batch_size:
                                (index + 1) * batch_size]})

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    print "starting training..."
    for epoch in xrange(training_epochs):
        # go through training set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_ae(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)
    print "training completed in : ", training_time


def epsilonNeighbor(x_scipySparse):
    pass


def test_epsilonNeighbor():
    x_scipySparse = None; train_set_x = None; numInstances = 0; numFeatures = 0;
    if((os.path.exists("input_scipySparse.obj"))):
        print "loading sparse data from pickled file..."
        f = open("input_scipySparse.obj", 'r')
        x_scipySparse = cPickle.load(f)
        f.close()
        numInstances, numFeatures = x_scipySparse.shape
        
    else: 
        print "extracting features and building sparse data..."
        fe = FeatureExtractor()  
        fe.extractFeatures()
        train_set_x = fe.instanceList
        featureDict = fe.featDict   
        numInstances = len(train_set_x)
        numFeatures = len(featureDict)        
        x_lil = sp.lil_matrix((numInstances,numFeatures), dtype='float32') # the data is presented as a sparse matrix 
        i = -1; v = -1;
        try:
            for i,instance in enumerate(train_set_x):
                for v in instance.input:
                    x_lil[i, v] = 1
        except:
            print "i=",i," v=",v
        x_scipySparse = x_lil.tocsc()
        f = open("input_scipySparse.obj", 'w')
        cPickle.dump(x_scipySparse, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    epsilonNeighbor(x_scipySparse)

if __name__ == '__main__':
    #test_AutoEncoder()
    test_epsilonNeighbor()
    