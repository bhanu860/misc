import data_io
from sklearn.ensemble import RandomForestClassifier
import cPickle
import os


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
    target = [0 for x in range(len(features_deleted))] + [1 for x in range(len(features_conf))]

    print("Training the Classifier")
    classifier = RandomForestClassifier(n_estimators=100, 
                                        verbose=2,
                                        n_jobs=1,
                                        min_samples_split=10,
                                        random_state=1,
                                        max_features=None)
    classifier.fit(features, target)
    
    print("Saving the classifier")
    data_io.save_model(classifier, prefix="forest_")
   
    
    
if __name__=="__main__":
    main()