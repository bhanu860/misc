'''
Created on May 3, 2013

@author: bhanu
'''

import csv
from Instance import Instance
import cPickle
import sys


class FeatureExtractor(object):
    '''
    classdocs
    '''

    featDict = dict()
    dup_dict = dict()  
    instanceList = []
    id_counter = -1
    author_paperDict = dict()
    paper_authorDict = dict()
    authors_noPapers = 0
    
  
    def extractFeatures(self):
    
        print "extracting features..."
        self.buildDicts()
        print "reading Author.csv..."
        with open('/home/bhanu/Downloads/dataRev2/Author.csv', 'rb') as csvfile:
            authorReader = csv.reader(csvfile)
            next(authorReader)
            
            #start reading authors and build their features        
            for row in authorReader:
                instance = Instance() 
                authorid = row[0]
                instance.id = authorid
                     
                #extract feature words corresponding to author name
                namestr = row[1].split(' ')
                for w in namestr :                    
                    pw = self.preprocess(w)
                    wid = self.getFeatureId(pw)
                    instance.name.add(wid)
              
                #extract keywords corresponding to author's affiliation
                for string in row[2:]:
                    nwords = string.split(' ')
                    for w in nwords:
                        pw = self.preprocess(w)
                        wid = self.getFeatureId(pw)
                        instance.affiliations.add(wid)
                        
                #extract features corresponding to co-authors of this author
                
                paperids = self.author_paperDict.get(authorid)
                if(paperids != None):
                    for pid in paperids:
                        for a in self.paper_authorDict[pid]: #add all authors to co-authors set
                            aid = self.getFeatureId("author"+a)
                            instance.co_authors.add(aid)
                    instance.co_authors.remove(self.getFeatureId("author"+authorid))            
                #what to do when author has no papers
                else:
                    instance.hasNoPapers = True  #data may be missing then
                    self.authors_noPapers +=1
  
                self.instanceList.append(instance)
                
        with open('instanceList.obj', 'w') as dumpfile:
            cPickle.dump(self.instanceList, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
        with open('featureDict.obj', 'w') as dumpfile:
            cPickle.dump(self.featDict, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
        
        print "finished extracting features."
        
                    
    def buildDicts(self):
        print "reading PaperAuthor.csv..."
        with open('/home/bhanu/Downloads/dataRev2/PaperAuthor.csv') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
          
            for row in csvreader:
                authorid = row[1]; paperid = row[0]
                paperids = self.author_paperDict.get(authorid)
                if(paperids == None):
                    self.author_paperDict[authorid] = []
                self.author_paperDict[authorid].append(paperid)
                authorids = self.paper_authorDict.get(paperid)
                if(authorids == None):
                    self.paper_authorDict[paperid] = []
                self.paper_authorDict[paperid].append(authorid)
  
    def getFeatureId(self, feature):
        wid = self.featDict.get(feature)
        if(wid == None):
            self.id_counter += 1
            self.featDict[feature] = self.id_counter
            wid = self.id_counter
        return wid
    
    
   
    
    def getScore(self, instance1, instance2):
        score = 0.0
        name = len(instance1.name) - len(instance1.name - instance2.name)
        if(name >= 2):     #score high for name match
            score += 0.5
        else:
            return 0.0
            
        aff = len(instance1.affiliations) - len(instance1.affiliations - instance2.affiliations)
        co_authors = len(instance1.co_authors) - len(instance1.co_authors - instance2.co_authors)
        commonPaper = self.featDict.get("author"+instance1.id) in instance2.co_authors
        
        if(aff >= 2):
            score += 0.25
        if(co_authors >= 1):
            score += 0.25
        if(commonPaper):
            score -= 0.5
        
        return score
    
  
    def rolf(self):
        print "running rolf..."
        for e in self.instanceList:
            self.dup_dict[e.id] = []
            self.dup_dict.get(e.id).append(e.id)
       
        try: 
            for indx, i in enumerate(self.instanceList[0:-2]):                
                for j in self.instanceList[indx+1:]:
                    if(self.getScore(i,j) >= 0.75):
                        self.dup_dict.get(i.id).append(j.id)
                        self.dup_dict.get(j.id).append(i.id)                
        except:
            print "unexpected error occurred : ", sys.exc_info()[0]
        finally:
            with open("dup-dict.obj", 'w') as dumpfile:
                cPickle.dump(self.dup_dict, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
            f = open("submission-file.txt", 'w')
            for a, values in  self.dup_dict.values():
                f.write(a+" ")
                for v in values:
                    f.write(v+" ")
                f.write("\n")
            f.flush()
            f.close()
            
    def load_data(self):
        print "loading instance list..."
        with open("instanceList.obj", 'r') as loadfile:
            self.instanceList = cPickle.load(loadfile)
        with open("featureDict.obj", 'r') as loadfile:
            self.featDict = cPickle.load(loadfile)
        
    
    def load_dup_dict(self):
        with open("dup-dict.obj", 'r') as loadfile:
            dup_dict = cPickle.load(loadfile)
        
        with open("submission-file.txt", 'w') as subfile:
            subfile.write("AuthorId,DuplicateAuthorIds\n") 
            for aid in dup_dict.keys():
                subfile.write(aid+',')
                for did in dup_dict.get(aid):
                    subfile.write(did+" ")
                subfile.write("\n")  
        
            
    def preprocess(self, w):
        if(w == None or w==''):
            pw = "None"
        else :
            pw = w.strip().strip(',').strip("")
            
        return pw
        
    def __init__(self):
        '''
        Constructor
        '''
        