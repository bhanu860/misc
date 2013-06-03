import data_io
from sklearn.ensemble import RandomForestClassifier
import cPickle
import os





def main():
    print("Getting features for each bird-class ")
    if(os.path.exists("features.obj")):
        with open("features.obj", 'r') as loadfile:
            features = cPickle.load(loadfile)
    else:
        features = data_io.get_features_csv()
        with open("features.obj", 'w') as dumpfile:
            cPickle.dump(features, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)

    target = []

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