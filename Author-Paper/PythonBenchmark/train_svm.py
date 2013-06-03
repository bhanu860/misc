import data_io
from sklearn import svm, cross_validation
import os
import cPickle
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm.classes import SVC
from sklearn.metrics.metrics import classification_report
from sklearn.cross_validation import train_test_split

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
    
    
    featuresnp = np.array(features, dtype='float32')
    targetnp = np.array(target, dtype='int32')   
    
   
    featuresnp -= np.mean(featuresnp, axis=0)
    featuresnp /= np.std(featuresnp, axis=0)
    
    # Set the parameters by cross-validation
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(featuresnp, targetnp, test_size=0.3, random_state=0)
    
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    scores = ['precision', 'recall']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=4, score_func=score, n_jobs=4, verbose=2)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.cv_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

    
    
    

#    print "Training svm model"
#    #clf  = svm.SVC(verbose=True, probability=True,)
#    clf  = svm.SVC(verbose=True)
#   
##    cv = cross_validation.KFold(len(features), n_folds=4)
#    cv = cross_validation.ShuffleSplit(len(features), n_iter=4, test_size=0.3, random_state=0)
#    results = cross_validation.cross_val_score(clf, X=featuresnp, y=targetnp, cv=cv, n_jobs=4, verbose=True)
#    #print out the mean of the cross-validated results
#    print "Results: ", results
#    print "Results: " + str( np.array(results).mean())
    
#    clf.fit(features, target)
#    print "saving linear logistic regression model"
#    data_io.save_model(clf, prefix="svm_")
    
if __name__=="__main__":
    main()