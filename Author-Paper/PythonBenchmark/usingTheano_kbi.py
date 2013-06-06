import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
import data_io
import time
import sys
import os
import cPickle
from sklearn import cross_validation

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                 dtype='float64'),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype='float64'),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        #print y.shape[0]
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def get_predictions(self):
        #return self.y_pred_shared.get_value()
        return self.y_pred#theano.shared(self.y_pred, borrow=True)
    
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.dtype, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype='float64')
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype='float64')
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out, n_hidden2=20, n_hidden3=10):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will
        # translate into a TanhLayer connected to the LogisticRegression
        # layer; this can be replaced by a SigmoidalLayer, or a layer
        # implementing any other nonlinearity
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)
        
        self.hiddenLayer2 = HiddenLayer(rng=rng, input=self.hiddenLayer.output,
                                       n_in=n_hidden, n_out=n_hidden2,
                                       activation=T.tanh)
        
        self.hiddenLayer3 = HiddenLayer(rng=rng, input=self.hiddenLayer2.output,
                                       n_in=n_hidden2, n_out=n_hidden3,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer3.output,
            n_in=n_hidden3,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer3.W).sum() \
                + abs(self.logRegressionLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer3.W ** 2).sum() \
                    + (self.logRegressionLayer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.hiddenLayer2.params + self.hiddenLayer3.params+ self.logRegressionLayer.params
#        self.params = self.hiddenLayer.params +  self.logRegressionLayer.params
        self.predictions = self.logRegressionLayer.get_predictions



def test_mlp(learning_rate=0.1, L1_reg=0.0001, L2_reg=0.0003, n_epochs=10000,
                          n_hidden=50):
        """
    
        :type learning_rate: float
        :param learning_rate: learning rate used (factor for the stochastic
        gradient
    
        :type L1_reg: float
        :param L1_reg: L1-norm's weight when added to the cost (see
        regularization)
    
        :type L2_reg: float
        :param L2_reg: L2-norm's weight when added to the cost (see
        regularization)
    
        :type n_epochs: int
        :param n_epochs: maximal number of epochs to run the optimizer
    
    
       """
        np.random.seed(17)  
        print("Getting features for deleted papers from the database")
        features_deleted = None; features_conf = None
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
                
                
#        print("Getting features for valid papers from the database")
#        if(os.path.exists("features_valid.obj")):
#            with open("features_valid.obj", 'r') as loadfile:
#                data = cPickle.load(loadfile)
#        else:
#            data = data_io.get_features_db("ValidPaper")
#            with open("features_valid.obj", 'w') as dumpfile:
#                cPickle.dump(data, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
        
                 
#        author_paper_ids = [x[:2] for x in data]
#        features_valid = [x[2:] for x in data]   
#        
#        features_validnp = np.array(features_valid, dtype='float64') 
        
#        predictInts = []
#        for tup in features_valid:
#           a, b, c, d, e = tup
#           predictInts.append((int(a), int(b), int(c), int(d), int(e)))
#      
#        predictsMat = np.ndarray(shape=(len(predictInts), 5), dtype='int32')
#        for i, tup in enumerate(predictInts):
#            a, b, c, d, e = tup
#            predictsMat[i, 0] = a;  predictsMat[i, 1] = b; predictsMat[i, 2] = c; predictsMat[i, 3] = d; predictsMat[i, 4] = e; 
#        predict_set_x = theano.shared(features_validnp, borrow=True)       
    
        features = [x[2:] for x in features_deleted + features_conf]
        target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]
        
        #code for including keywords match feature
        print "adding additional features..."
        import additional_features as af
        all_features = af.get_additional_features()    
        kw_deleted, kw_confirmed, _ = all_features
        kw_features = kw_deleted+kw_confirmed
        for i in range(len(features)):
            features[i]+= tuple(kw_features[i][2:])
        
        
        
        featuresnp = np.array(features, dtype='float64')
        targetnp = np.array(target, dtype='int32')
    
        featuresnp -=np.mean(featuresnp, axis=0)
        featuresnp /=np.std(featuresnp, axis=0)
 
            
        cv = cross_validation.ShuffleSplit(len(features), n_iter=1, test_size=0.25, random_state=0)
        for train, test in cv:
            train_set_x = theano.shared(featuresnp[train], borrow=True)
            test_set_x = theano.shared(featuresnp[test], borrow=True)
            train_set_y = theano.shared(targetnp[train], borrow=True)
            test_set_y=theano.shared(targetnp[test], borrow=True)
 
    
        batch_size = 20    # size of the minibatch

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
#        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size
    
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        # allocate symbolic variables for the data
        
#        size = T.lscalar()  
        index = T.lscalar()
        x = T.matrix('x', dtype='float64')  # sparse.csr_matrix('x', dtype='int32'); the data is presented as sparse matrix
        y = T.ivector('y')  # the labels are presented as 1D vector of
    
                            # [int] labels
    
        rng = np.random.RandomState(113)
        
        # construct the MLP class
        classifier = MLP(rng=rng, input=x, n_in=featuresnp.shape[1],
                         n_hidden=n_hidden, n_out=2)
    
        cost = classifier.negative_log_likelihood(y) \
             + L1_reg * classifier.L1 \
             + L2_reg * classifier.L2_sqr
    

        test_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})
 
#        predict_model = theano.function(inputs=[],
#                outputs=classifier.predictions(),
#                givens={
#                    x: predict_set_x})
    
        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in classifier.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
    
        # specify how to update the parameters of the model as a dictionary
        updates = OrderedDict()
        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        for param, gparam in zip(classifier.params, gparams):
            updates[param] = param - learning_rate * gparam
 
        train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    
        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        

        # early-stopping parameters
        patience = 1000000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.0995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
        best_params = None
        best_validation_loss = np.inf
        best_iter = 0
        test_score = 0.
        start_time = time.clock()
    
        epoch = 0
        done_looping = False
    
        best_params = None
        while True:

            try :
                
                epoch = epoch + 1
                training_cost = []
                for minibatch_index in xrange(n_train_batches):
                    minibatch_avg_cost = train_model(minibatch_index)
                    training_cost.append(minibatch_avg_cost)
                    # iteration number
                    iter = (epoch - 1) * n_train_batches + minibatch_index
        
                    if (iter + 1) % validation_frequency == 0:
                        # compute zero-one loss on validation set
                        validation_losses = [test_model(i) for i
                                             in xrange(n_test_batches)]
                        this_validation_loss = np.mean(validation_losses)
        
                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                             (epoch, minibatch_index + 1, n_train_batches,
                              this_validation_loss * 100.))
        
                        # if we got the best validation score until now
                        if this_validation_loss < best_validation_loss:
                            #improve patience if loss improvement is good enough
                            if this_validation_loss < best_validation_loss *  \
                                   improvement_threshold:
                                patience = max(patience, iter * patience_increase)
        
                            best_validation_loss = this_validation_loss
                            best_iter = iter
                            best_params = classifier.params
        
                mean_cost = np.mean(training_cost)
                print "Epoch ", epoch," training cost: ", mean_cost
                
            except KeyboardInterrupt:
                print "Training ended by user.\n"
#                #update params one last time in case we interrupted the training in middle of updates
#                for minibatch_index in xrange(n_train_batches):
#                    train_model(minibatch_index)
                print "Best Validation loss:", best_validation_loss                
                break 
                
       
    
        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
               'obtained at iteration %i, with test performance %f %%') %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        
        print("Saving the mlp best params")
        data_io.save_model(best_params, prefix="theano_")
        
        ############################
        #Making Predictions
        ############################
        
#        print("Making predictions")
#        predictions = predict_model()#classifier.predict_proba(features_valid)[:,1]
#        predictions = list(predictions)
#    
#        author_predictions = defaultdict(list)
#        paper_predictions = {}
    
#        for (a_id, p_id), pred in zip(author_paper_ids, predictions):
#            author_predictions[a_id].append((pred, p_id))
#    
#        for author_id in sorted(author_predictions):
#            paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
#            paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]
#    
#        print("Writing predictions to file")
#        data_io.write_submission(paper_predictions, prefix="theano_")
        

if __name__ == '__main__':
    test_mlp()