from collections import defaultdict
import data_io
import cPickle
import os
import numpy as np
import math
from train_mlp import Mlp, Layer

def main():
    
    print("Getting features for valid papers from the database")
    if(os.path.exists("features_valid.obj")):
        with open("features_valid.obj", 'r') as loadfile:
            data = cPickle.load(loadfile)
    else:
        data = data_io.get_features_db("ValidPaper")
        with open("features_valid.obj", 'w') as dumpfile:
            cPickle.dump(data, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
    
    author_paper_ids = [x[:2] for x in data]
    features = [x[2:] for x in data]
    
    predictInts = []
    for tup in features:
        a, b, c, d, e = tup
        predictInts.append((int(a), int(b), int(c), int(d), int(e)))

    print("Loading the classifier")
    mlp = data_io.load_model(prefix="mlp_")

    print("Making predictions")
    predictions = []
    for x in predictInts : 
        #Propagate the inputs forward to compute the outputs             
        outp = list(x)     #output of  input layer i.e. output of previous layer to be used as input for next layer
        for layer in mlp.layers[1:] :           #for all layers starting from the second layer
            for i in range(layer.nNeurons):
                layer.net[i] =  weightedSum(outp, layer.W[1:,i]) + layer.W[0,i]
                layer.out[i] = g(layer.net[i], layer.transferF)   #pass this weighted sum through the transfer function of this layer                  
                outp = layer.out  
        predictions.append(mlp.layers[-1].out[0])

    author_predictions = defaultdict(list)
    paper_predictions = {}

    for (a_id, p_id), pred in zip(author_paper_ids, predictions):
        author_predictions[a_id].append((pred, p_id))

    for author_id in sorted(author_predictions):
        paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
        paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]

    print("Writing predictions to file")
    data_io.write_submission(paper_predictions, prefix="mlp_")




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

def weightedSum(inputVector, weights):
#        print inputVector
#        print weights
    sum = (np.sum(inputVector*weights)) 
#        print sum
    return sum

if __name__=="__main__":
    main()