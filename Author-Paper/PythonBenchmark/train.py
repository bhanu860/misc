import data_io
from sklearn.ensemble import RandomForestClassifier
import cPickle
import os
import numpy as np
from theano.gof.python25 import defaultdict
from sklearn import cross_validation




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
    all_features = af.get_additional_features()    
    kw_deleted, kw_confirmed, _ = all_features
    kw_features = kw_deleted+kw_confirmed
    for i in range(len(features)):
        features[i]+= tuple(kw_features[i][2:])
 
 
    #Simple K-Fold cross validation. 10 folds.
    #cv = cross_validation.KFold(len(features), n_folds=5)
    cv = cross_validation.ShuffleSplit(len(features), n_iter=4, test_size=0.4, random_state=0)
    
    print("Training the Classifier")
    classifier = RandomForestClassifier(n_estimators=100, 
                                        verbose=2,
                                        n_jobs=1,
                                        min_samples_split=1,
                                        random_state=0, 
                                        compute_importances=True                                        
                                        )
 

    featuresnp = np.array(features, dtype='int32')
    targetnp = np.array(target, dtype='int32')
    
#    with open("wrong_predictions.txt", 'w' ) as wp:
#        class1count = 0; class2count =0; rpredictions = 0
#        for train, test in cv:
#            x_train = featuresnp[train];        y_train = targetnp[train]
#            x_test = featuresnp[test];         y_test = targetnp[test]
#            classifier.fit(x_train, y_train)
#            predictions = classifier.predict_proba(x_test)
#            pred_classes = classifier.predict(x_test)
#            for i in range(len(y_test)):
#            
#                if y_test[i] != pred_classes[i] :
#                    if(predictions[i,0] > 0.5 and predictions[i,0] < 0.6):
#                        class1count+=1;
#                    if(predictions[i,1] > 0.5 and predictions[i,1] < 0.6):
#                        class2count+=1;
#                    line = "feat: "+str(features[test[i]])+" ".join([ " a:",str(y_test[i])," p:", str(pred_classes[i])," proba:", str(predictions[i]), "\n"])
#                    wp.write(line)
#                else:
#                    if(predictions[i,0] > 0.4 and predictions[i,0] < 0.6):
#                        rpredictions+=1;
#                    
#        print "number of wrong predictions of deleted class: ", class1count
#        print "number of wrong predictions of confirmed class: ", class2count
#        print "number of right predictions with close probas", rpredictions
#        for train, test in cv:
#            print "total number of test examples: ", len(test)
        

#    classifier.fit(featuresnp, targetnp)
#    importances = classifier.feature_importances_
##    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
##                 axis=0)
#    indices = np.argsort(importances)[::-1]
#    
#    # Print the feature ranking
#    print("Feature ranking:")
#    
#    for f in range(len(indices)):
#        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#    numFeatures = 15   
#    prunedFeatures = np.zeros(shape=(featuresnp.shape[0], numFeatures), dtype="int32")
#    for i in range(prunedFeatures.shape[0]):
#        for j, fi in enumerate(indices[0:numFeatures]):
#            prunedFeatures[i,j] = featuresnp[i, fi]  
            

#    featuresnp -= np.mean(featuresnp, axis=0)
#    featuresnp /= np.std(featuresnp, axis=0)

#       
    results = cross_validation.cross_val_score(classifier, X=featuresnp, y=targetnp, cv=cv, n_jobs=4, verbose=True)
    #print out the mean of the cross-validated results
    print "Results: ", results
    print "Results: " + str( np.array(results).mean())
#    
    
#    print "training...."
#    classifier.fit(featuresnp, targetnp)
#    print("Saving the classifier")
#    data_io.save_model(classifier, prefix="forest_")
   


####Making predictions######
#    print("Getting features for valid papers from the database")
#    if(os.path.exists("features_valid.obj")):
#        with open("features_valid.obj", 'r') as loadfile:
#            data = cPickle.load(loadfile)
#    else:
#        data = data_io.get_features_db("ValidPaper")
#        with open("features_valid.obj", 'w') as dumpfile:
#            cPickle.dump(data, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
#    author_paper_ids = [x[:2] for x in data]
#    features_valid = [x[2:] for x in data]
#    
#    #code for including keywords match feature
#    print "adding addtional features..."
#    all_features = af.get_additional_features()    
#    _, _, kw_features_valid = all_features    
#    for i in range(len(features_valid)):
#        features_valid[i]+= tuple(kw_features_valid[i][2:])
#    
#    features_validnp = np.array(features_valid, dtype='int32')
#        
##    featuresnp -= np.mean(featuresnp, axis=0)
##    featuresnp /= np.std(featuresnp, axis=0)
#    
#    
##    print("Loading the classifier")
##    classifier = data_io.load_model(prefix="forest_")
#
#    print("Making predictions")
#    predictions = classifier.predict_proba(features_validnp)[:,1]
#    predictions = list(predictions)
#
#    author_predictions = defaultdict(list)
#    paper_predictions = {}
#
#    for (a_id, p_id), pred in zip(author_paper_ids, predictions):
#        author_predictions[a_id].append((pred, p_id))
#
#    for author_id in sorted(author_predictions):
#        paper_ids_sorted = sorted(author_predictions[author_id], reverse=True)
#        paper_predictions[author_id] = [x[1] for x in paper_ids_sorted]
#
#    print("Writing predictions to file")
#    data_io.write_submission(paper_predictions, prefix="forest_")


    
    
if __name__=="__main__":
    main()