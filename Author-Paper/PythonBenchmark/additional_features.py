'''
Created on May 29, 2013

@author: bhanu
'''
import data_io
import re
import cPickle
import os

keywords_dict = dict()
author_keywords = dict()
author_titles = dict()
all_features = []
delimiters = " ", "|", "\\", ";" , ",",":" 
regexPattern = '|'.join(map(re.escape, delimiters))
tables = [ "TrainDeleted","TrainConfirmed", "ValidPaper"]  #order of tables is important.. should match the order in train.py

def build_keywords_dict():
    print "Building keywords dictonary for authors..."
    author_keywords_query = "select authorid, array_agg(keyword) from paperauthor left outer join paper on id=paperid group by authorid"
    authorkeywords = data_io.run_query_db(author_keywords_query)
    
    #for keywords
    for aid, keywords in authorkeywords :
        i = int(aid)
        if(author_keywords.get(i) ==  None):
            author_keywords[i] = dict()
        if(keywords == None):
            continue
        
        for kw in keywords:
            if(kw==None):
                continue
            w = re.split(regexPattern, kw) 
            for ww in w:
                if(ww==""):
                    continue
                addtoauthorsdict(i, ww)


def addtoauthorsdict(aid, word):
    d = author_keywords.get(aid)
    if(d.get(word)==None):
        d[word] = 1
    else:
        d[word] = d.get(word)+1


                
def build_titles_dict():
    print "building title dict for authors..."
    #author_title_query = "select authorid,  array_agg(title) from paperauthor left outer join paper on id=paperid group by authorid"
    #authortitles = data_io.run_query_db(author_title_query)
    #for titles
    n = len(author_keywords)
    authortitles = get_title_set()
    for aid, titles in authortitles :
        i = int(aid)
        if(author_titles.get(i) ==  None):
            author_titles[i] = set()
        if(titles == None):
            continue
        
        for kw in titles:
            if(kw==None):
                continue
            w = re.split(regexPattern, kw) 
            for ww in w:
                if(ww==""):
                    continue
                author_titles.get(i).add(ww)
        
    print "finished building author-keywords dictionary."
            

def get_title_set(authorid):
    aids =""
    for aid in authorid:
        aids += str(aid)+","
    query = "select authorid, array_agg(title) from paperauthor left outer join paper on id=paperid where authorid in ("+aids+") group by authorid"
    authortitles = data_io.run_query_db(query)
#    titles_set = set()
#    for (aid, titles) in authortitles:
#        if(titles == None):
#            continue        
#        for kw in titles:
#            if(kw==None):
#                continue
#            w = re.split(regexPattern, kw) 
#            for ww in w:
#                if(ww==""):
#                    continue
#                titles_set.add(ww)
    return authortitles
            
     
def build_keywords_feature():    
    build_keywords_dict()
    global all_features 
    for datatable in tables:
        print "building features for "+datatable+"..."
        query = "select authorid, paperid, keyword from "+datatable+" left outer join paper on id=paperid order by authorid, paperid"
        apkeywords = data_io.run_query_db(query)
        features = []
        for aid, pid, keywords in apkeywords:
            #titles_set = get_title_set(int(aid))
            words = set()
            #keywords += title
            if(keywords==None or keywords==""):
                features.append((aid, pid, 0))
                continue            
            w = re.split(regexPattern, keywords)
            for ww in w:
                if(ww==""):
                    continue
                words.add(ww)
            kwfreq = 0
            for w in words:
                count = author_keywords.get(aid).get(w)
                if(count == None):
                    count = 0
                kwfreq += count
            
            norm = len(words)
            if(norm == 0):
                norm = 1 
            kwfreq = kwfreq#/(norm)   
#            kwntitles = author_keywords.get(aid) | titles_set
#            ncommon_words = (len(kwntitles) - len(kwntitles - words)) / (len(kwntitles)+1.0)
            print aid, " ", pid, " ", kwfreq
            features.append((aid, pid, kwfreq))
        all_features.append(features)
    with open("keyword_features.obj", 'w') as dumpfile:
        cPickle.dump(all_features, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)
    

def get_keywords_feature():
    global all_features
    if(os.path.exists("keyword_features.obj")):
        print "Loading keyword feature..."
        with open("keyword_features.obj", 'r') as loadfile:
            all_features = cPickle.load(loadfile)
    else:    
        build_keywords_feature()
        
    return all_features          
                
                 
if __name__=="__main__":
    #get_keywords_feature()
    build_titles_dict()
    
    
    
    
    
    
    
    
    
    
    