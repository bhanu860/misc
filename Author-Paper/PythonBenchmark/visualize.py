import pylab as pl
import numpy as np

from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.decomposition import RandomizedPCA
from sklearn.naive_bayes import BernoulliNB
import os
import cPickle
import data_io
from sklearn import manifold



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
    
    

#featuresnp = np.array(features[0:2000]+features[-2000:], dtype='float32')
#targetnp = np.array(target[0:2000]+target[-2000:], dtype='int32')

featuresnp = np.array(features, dtype='float32')
targetnp = np.array(target, dtype='int32')

featuresnp -= np.mean(featuresnp, axis=0)
featuresnp /= np.std(featuresnp, axis=0)


# make a synthetic dataset
X, y = featuresnp, targetnp

# use RandomTreesEmbedding to transform data
hasher = RandomTreesEmbedding(n_estimators=50, random_state=0, max_depth=1)
X_transformed = hasher.fit_transform(X)

## Visualize result using PCA
#pca = RandomizedPCA(n_components=50)
#X_reduced = pca.fit_transform(X_transformed)

print("Computing Isomap embedding")

X_reduced = manifold.Isomap(n_neighbors=30, n_components=2).fit_transform(X)
print("Done.")

#print("Computing Spectral embedding")
#embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                      eigen_solver="arpack")
#X_reduced = embedder.fit_transform(X)

## Learn a Naive Bayes classifier on the transformed data
#nb = BernoulliNB()
#nb.fit(X_transformed, y)


# Learn an ExtraTreesClassifier for comparison
#trees = ExtraTreesClassifier(max_depth=3, n_estimators=100, random_state=0)
#trees.fit(X, y)


# scatter plot of original and reduced data
fig = pl.figure()

ax = pl.subplot(221)
ax.scatter(X[:, 0], X[:, 1], c=y, s=10)
ax.set_title("Original Data (2d)")
ax.set_xticks(())
ax.set_yticks(())

ax = pl.subplot(222)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, s=10)
ax.set_title("PCA reduction (2d) of transformed data (%dd)" %
             X_transformed.shape[1])
ax.set_xticks(())
ax.set_yticks(())

# Plot the decision in original space. For that, we will assign a color to each
# point in the mesh [x_min, m_max] x [y_min, y_max].
h = .01
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

## transform grid using RandomTreesEmbedding
#transformed_grid = hasher.transform(np.c_[xx.ravel(), yy.ravel()])
#y_grid_pred = nb.predict_proba(transformed_grid)[:, 1]
#
#ax = pl.subplot(223)
#ax.set_title("Naive Bayes on Transformed data")
#ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
#ax.scatter(X[:, 0], X[:, 1], c=y, s=50)
#ax.set_ylim(-1.4, 1.4)
#ax.set_xlim(-1.4, 1.4)
#ax.set_xticks(())
#ax.set_yticks(())

## transform grid using ExtraTreesClassifier
#y_grid_pred = trees.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#
#ax = pl.subplot(224)
#ax.set_title("ExtraTrees predictions")
#ax.pcolormesh(xx, yy, y_grid_pred.reshape(xx.shape))
#ax.scatter(X[:, 0], X[:, 1], c=y, s=50)
#ax.set_ylim(-1.4, 1.4)
#ax.set_xlim(-1.4, 1.4)
#ax.set_xticks(())
#ax.set_yticks(())

pl.tight_layout()
pl.show()