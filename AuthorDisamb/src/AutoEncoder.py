'''
Created on May 5, 2013

@author: bhanu
'''

import theano
import numpy
import theano.tensor as T
from theano import sparse
import scipy.sparse as sp

class AutoEncoder(object):
    '''
    classdocs
    '''
    
    def __init__(self, numpy_rng, input=None, n_visible=784, n_hidden=500,
           W=None, bhid=None, bvis=None, n_trainExs=0):
        """
    
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights
    
    
        :type input: theano.tensor.TensorType
        :paran input: a symbolic description of the input or None for standalone
                      dA
    
        :type n_visible: int
        :param n_visible: number of visible units
    
        :type n_hidden: int
        :param n_hidden:  number oMyClassf hidden units
    
        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None
    
        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
    
        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
    
    
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_trainExs = n_trainExs
    
    
        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and 4*sqrt(6./(n_hidden+n_visible))
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype='float32')
            W = theano.shared(value=initial_W, name='W')
    
        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible,
                                        dtype='float32'), name='bvis')
    
        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden,
                                              dtype='float32'), name='bhid')
    
    
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        # if no input is given, generate a variable representing the input
        if input == None:
            # we use a matrix because we expect a minibatch of several examples,
            # each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
    
        self.params = [self.W, self.b, self.b_prime]
        
        
    def get_hidden_values(self, inp):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(sparse.dot(inp, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """ Computes the reconstructed input given the values of the hidden layer """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
    
    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the Auto Encoder """

        #tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch      
    
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of thefile
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
    
    def get_cost_updates1(self, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the Auto Encoder """

        #tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch      
          
        L = T.sum()
        #L = - T.sum(a + b * c, axis=1)
        #L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        #L = - T.sum(self.x * T.log(z) + tildx * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of thefile
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)
                