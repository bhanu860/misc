import data_io
from sklearn.ensemble import RandomForestClassifier
import cPickle
import os
import numpy as np
from sklearn import cross_validation

from sklearn.ensemble import GradientBoostingClassifier


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
    
    
    #code for including keywords match feature
    print "adding addtional features..."
    import additional_features as af
    all_features = af.get_keywords_feature()
    kw_deleted, kw_confirmed, _ = all_features
    kw_features = kw_deleted+kw_confirmed
    for i in range(len(features)):
        _,_,ckw = kw_features[i]
        features[i]+=(ckw,)
 
 
    #Simple K-Fold cross validation. 10 folds.
    #cv = cross_validation.KFold(len(features), n_folds=5)
    cv = cross_validation.ShuffleSplit(len(features), n_iter=4, test_size=0.3, random_state=0)
    
    print("Training the Classifier")
    classifier = RandomForestClassifier(n_estimators=200, 
                                        verbose=2,
                                        n_jobs=1,
                                        min_samples_split=5,
                                        random_state=0)
#    classifier = GradientBoostingClassifier(n_estimators=200, 
#                                        verbose=2,
#                                        min_samples_split=5,
#                                        random_state=0)
#    

    featuresnp = np.array(features, dtype='float32')
    targetnp = np.array(target, dtype='int32')
    
    featuresnp -= np.mean(featuresnp, axis=0)
    featuresnp /= np.std(featuresnp, axis=0)
    
    results = cross_validation.cross_val_score(classifier, X=featuresnp, y=targetnp, cv=cv, n_jobs=4, verbose=True)
    #print out the mean of the cross-validated results
    print "Results: ", results
    print "Results: " + str( np.array(results).mean())
#    
    
    
#    importances = classifier.feature_importances_
##    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],axis=0)
#    indices = np.argsort(importances)[::-1]
#
#    # Print the feature ranking
#    print("Feature ranking:")
#
#    for f in range(10):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    
#    classifier.fit(featuresnp, targetnp)
#    print("Saving the classifier")
#    data_io.save_model(classifier, prefix="forest_")
   
    
    
if __name__=="__main__":
    main()