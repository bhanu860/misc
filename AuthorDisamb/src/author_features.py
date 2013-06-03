'''
Created on May 29, 2013

@author: bhanu
'''
import data_io
import re
import cPickle
import os
from sklearn.cluster.k_means_ import MiniBatchKMeans
import numpy as np
import pylab as pl
from sklearn.metrics.pairwise import euclidean_distances
import scipy 


author_name = []

delimiters = " ", "|", "\\", ";" , ",",":",".","-" 
regexPattern = '|'.join(map(re.escape, delimiters))

def build_author_name_dict():
    print "Building name dictonary for authors..."
    author_name_query = "select id, name from author"
    authorname = data_io.run_query_db(author_name_query)
    for aid, name in authorname:
        words = set()
        if(name == None or name==""):
            author_name.append((aid, words))
            continue
        w = re.split(regexPattern, name) 
        for ww in w:
            if(ww==""):
                continue
            words.add(ww)
        author_name.append((aid, words))
    print "finished building author-name dictionary."
    return author_name
            


def build_training_data():
    f = open("training.dat", 'w') 
    for id1, name1 in author_name[0:100]:
        for id2, name2  in author_name:
            id_match = 1.0 if id1 == id2 else 0.0
            unionset = name1|name2
            intersectionset = name1 & name2
            d = len(unionset)+0.0 if(len(unionset) != 0) else 1.0
            name_match = len(intersectionset) / d
            y = 0
            if(id_match == 1):
                y = 1
            if(name_match == 1):
                y = 1
#            if(name_match >= 0.5 and ):
                
            f.write(str(id_match)+" "+str(name_match)+"    ")
            print id_match, name_match


def extract_data(start, end):
    print "extracting clustering data from authors: ", str(start)," to ", str(end)
    #total number of samples = nofauthors^2, training sample = num of authors with same name or same id i.e. nauthors+samenameauthors
    #same id or same name qualifies as duplicate author
    # X :- x1: Id match, x2: fraction of Name match
    data = []

    for id1, name1 in author_name[start:end]:
        for id2, name2  in author_name[start+1:]:
            id_match = 1.0 if id1 == id2 else 0.0
            unionset = name1|name2
            intersectionset = name1 & name2
            d = len(unionset)+0.0 if(len(unionset) != 0) else 1.0
            name_match = len(intersectionset) / d
            
            print id_match, name_match
            data.append([id_match, name_match])
    return np.array(data)
    
    

def cluster_kmeans(n_samples):    
    rng = np.random.RandomState(9)
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=rng, verbose=True, compute_labels=True)
    i = 0; 
    batch_size = 10
    
    while(i < n_samples):
        #partial fit 100 authors and there subsequent comparisons  
        print "k_means partial fitting, i = ", str(i)      
        data = extract_data(start=i, end=i+batch_size)
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)
        kmeans.partial_fit(data)
        i+=batch_size     
    print "fitting of one-third data finished."
    return kmeans
    


def train():
    if (os.path.exists("k_means.obj")) :
        with open("k_means.obj", 'r') as loadfile:
            kmeans = cPickle.load(loadfile)   
    else:
        n_authors = len(author_name)
        n_samples = 100 #n_authors/3
        kmeans = cluster_kmeans(n_samples)
        print "Saving kmeans model..."
        with open("k_means.obj", 'w') as dumpfile:
            cPickle.dump(kmeans, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
            

def predict():
    print "loading kmeans classfier..."
    with open("k_means.obj", 'r') as loadfile:
            kmeans = cPickle.load(loadfile) 
    
    batch_size = 1
    i = 0
    n = len(author_name)
    res = scipy.sparse.csc_matrix((n,n), dtype="int32")
    while (i < n):        
        data = extract_data(i, i+batch_size)
        Y = kmeans.predict(data)
        i+=batch_size
        for j,v in enumerate(Y):
            if(v != 0):
                print author_name[i], "equals", author_name[i+j]
                res[i,i+j] = 1


def same_words(str1, str2): 
    import re
    delimiters = " ",".","-" 
    regexPattern = '|'.join(map(re.escape, delimiters))
    string1 = str1.strip().lower()
    string2 = str2.strip().lower()
    
    nstr1 = re.split(regexPattern, string1)
    nstr2 = re.split(regexPattern, string2)
    if(len(nstr1) <=1 or len(nstr2) <= 1):
        return False
    count1 = 0; count2 = 0
    for w in nstr1:
        if w in set(nstr2):
            count1+=1
    if count1==len(nstr1):
        return True        
    for w in nstr2:
        if w in set(nstr1):
            count2+=1
    if count2==len(nstr2):
        return True
    
    return False
            
       
                
if __name__=="__main__":
#    build_author_name_dict()
#    #train()
#    predict()
    print same_words("hello World is", "Hello is")
    
    
    
    
    
    
    
    
    
    
    