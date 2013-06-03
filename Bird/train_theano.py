
import numpy as np
import theano
import theano.tensor as T
import data_io
import time
import sys
import os
from Mlp import MLP
import cPickle




def load_data(features, target):
   
    trainend = 3*len(features)/4 
    testend = trainend+ len(features)/8
    
#    train_set_x, test_set_x, valid_set_x = features[0:trainend], features[trainend:testend], features[testend:]
#    
#    train_set_y, test_set_y, valid_set_y = target[0:trainend], target[trainend:testend], target[testend:]

    train_set_x, test_set_x, valid_set_x = features[0:-32], features[-32:-16], features[-16:]
    
    train_set_y, test_set_y, valid_set_y = target[0:-32], target[-32:-16], target[-16:]
    
    def shared_dataset(data_x, data_y):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        #data_x, data_y = data_xy
        shared_x = theano.shared(data_x, borrow=True)
        shared_y = theano.shared(data_y, borrow=True)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, shared_y #T.cast(shared_y, 'int32')

    test_set_x,  test_set_y  = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x,valid_set_y), (test_set_x, test_set_y)]
    return rval


def test_mlp(learning_rate=0.013, L1_reg=0.00, L2_reg=0.0003, n_epochs=300,
                          n_hidden=100):
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
        np.random.seed(0)  
        print("Getting features for bird classes")
        if(os.path.exists("features.obj")):
            with open("features.obj", 'r') as loadfile:
                features, target = cPickle.load(loadfile)
        else:
            features, target = data_io.get_features_mat()
            with open("features.obj", 'w') as dumpfile:
                cPickle.dump((features, target), dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
    
            
        datasets = load_data(features, target)#gen_data2()
    
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]
        
    
        batch_size = 16    # size of the minibatch

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches  = test_set_x.get_value(borrow=True).shape[0]  / batch_size
    
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'
    
        # allocate symbolic variables for the data
        
        index = T.lscalar()
        x = T.matrix('x', dtype='float64')  # sparse.csr_matrix('x', dtype='int32'); the data is presented as sparse matrix
        y = T.ivector('y')  # the labels are presented as 1D vector of
    
                            # [int] labels
    
        rng = np.random.RandomState(113)
        
        # construct the MLP class
        classifier = MLP(rng=rng, input=x, n_in=features.shape[1],
                         n_hidden=n_hidden, n_out=35)
    
        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = classifier.negative_log_likelihood(y) \
             + L1_reg * classifier.L1 \
             + L2_reg * classifier.L2_sqr
        
    
        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
#        test_model = theano.function(inputs=[size],
#                outputs=[classifier.errors(y),classifier.getPredictions()],
#                 givens={
#                    x: test_set_x[0:size],
#                    y: test_set_y[0:size]}
#                )
#    
#        validate_model = theano.function(inputs=[size],
#                outputs=[classifier.errors(y),classifier.getPredictions()],
#                 givens={
#                    x:valid_set_x[0:size],
#                    y:valid_set_y[0:size]}
#                )

        test_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})
    
        validate_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                    y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
        
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
        updates = {}
        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        for param, gparam in zip(classifier.params, gparams):
            updates[param] = param - learning_rate * gparam
    
        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
#        train_model = theano.function(inputs=[size],
#                                      outputs=cost,
#                updates=updates,
#                givens={
#                    x: train_set_x[0:size],
#                    y: train_set_y[0:size]}
#               )
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
    
        
        
        
        while (epoch < n_epochs) and (not done_looping):
            #datasets = load_data(featuresMat, targetInts)#permute data()
#    
#            train_set_x, train_set_y = datasets[0]
#            valid_set_x, valid_set_y = datasets[1]
#            test_set_x, test_set_y = datasets[2]
            epoch = epoch + 1
            training_cost = []
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_index)
                training_cost.append(minibatch_avg_cost)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index
    
                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                         in xrange(n_valid_batches)]
                    this_validation_loss = np.mean(validation_losses)
    
#                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
#                         (epoch, minibatch_index + 1, n_train_batches,
#                          this_validation_loss * 100.))
    
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                               improvement_threshold:
                            patience = max(patience, iter * patience_increase)
    
                        best_validation_loss = this_validation_loss
                        best_iter = iter
                        best_params = []
                        best_params.append(classifier.params)
    
                        # test it on the test set
                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_score = np.mean(test_losses)
    
#                        print(('     epoch %i, minibatch %i/%i, test error of '
#                               'best model %f %%') %
#                              (epoch, minibatch_index + 1, n_train_batches,
#                               test_score * 100.))
#    
                mean_cost = np.mean(training_cost)
                if(mean_cost < 0.0005):
                    done_looping = True
                    print "training cost: ", mean_cost
                    break
            print "Epoch ", epoch," training cost: ", mean_cost
       
    
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
#    
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