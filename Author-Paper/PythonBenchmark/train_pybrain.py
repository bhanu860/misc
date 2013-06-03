'''
Created on May 16, 2013

@author: bhanu
'''
from pybrain.datasets import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
import cPickle
import data_io
import os

def train():
    ninp = 5
    nhidden = 10
    noutput = 1
    inpDim = 5
    targetDim = 1
    
    net = buildNetwork(ninp, nhidden, noutput, bias=True)
    ds = SupervisedDataSet(inpDim, targetDim)
    
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

    features = [x[2:] for x in features_deleted + features_conf]
    target = [[0] for x in range(len(features_deleted))] + [[1] for x in range(len(features_conf))]
    
    featuresInts = []
    for tup in features:
        a, b, c, d, e = tup
        featuresInts.append((int(a), int(b), int(c), int(d), int(e)))
    
    trainset = zip(featuresInts, target)
    
    for x, y in trainset:
        ds.addSample(x, y)
    
    print "training..."
    trainer = BackpropTrainer(net, ds)
    trainer.trainUntilConvergence()
    with open("net_pybrain.obj", 'w') as dumpfile:
        cPickle.dump(net, dumpfile, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    train()