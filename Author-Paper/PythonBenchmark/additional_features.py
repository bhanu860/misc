'''
Created on May 29, 2013

@author: bhanu
'''
import data_io
import re
import cPickle
import os


author_keywords = dict()
author_titles = dict()
all_features = None
delimiters = " ", "|", "\\", ";" , ",",":","-" 
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


def build_titles_feature():
    global author_titles, all_features        
    for i, datatable in enumerate(tables):
        print "building title features  for ", datatable
        author_title_query = "select t.authorid, t.paperid,  p.title from "+datatable+\
            " t  left outer join paper p on p.id=t.paperid order by t.authorid, t.paperid"
        authortitles = data_io.run_query_db(author_title_query)
        #build title dictionary for all authors
        for aid, pid, title in authortitles :            
            if(author_titles.get(aid) == None):
                author_titles[aid] = dict()
            for word in re.split(regexPattern, title):
                word = word.strip()
                if(len(word) > 3):  #eleminate all 2 letter words
                    if(author_titles.get(aid).get(word) == None):
                        author_titles.get(aid)[word] = 1
                    else:
                        author_titles.get(aid)[word] +=1
        
        title_feature = []
        for aid, pid, title in authortitles:
            totalauthorwords = len(author_titles.get(aid))
            totalpaperwords = len(re.split(regexPattern, title))
            commonwords = 0
            for word in re.split(regexPattern, title):
                key = word.strip()
                if key in author_titles.get(aid).keys():
                    commonwords +=author_titles.get(aid).get(key)
            title_feature.append((aid, pid, totalauthorwords, totalpaperwords, commonwords)) 
            
        this_table_feats = all_features[i]
        for ii,t in enumerate(title_feature): 
            this_table_feats[ii] +=t[2:]
         
    print "finished building title features."
            


def build_confnJour_feature():
    global all_features
    for i, datatable in enumerate(tables):
        print "building conference and journal features for "+datatable+"..."
#        query = "with confpapers as (select conferenceid, count(id) as confpapercount from paper group by conferenceid), "+\
#                "jourpapers as (select journalid, count(id) as jourpapercount from paper group by journalid)"+\
#                "select t.authorid, t.paperid, confpapercount, jourpapercount from "+datatable+" left outer join paper p on p.id=t.paperid, confpapers c, jourpapers j"+\
#                " where c.conferenceid=p.conferenceid and j.journalid=p.journalid order by authorid, paperid"
        query = """ with confpapers as (
                select conferenceid, count(id) as confpapercount 
                from paper 
                group by conferenceid),
            jourpapers as (
                select journalid, count(id) as jourpapercount 
                from paper 
                group by journalid)
            select t.authorid, t.paperid, confpapercount, jourpapercount 
            from """+datatable+""" t 
                left outer join paper p on p.id=t.paperid, 
                confpapers c, 
                jourpapers j
            where c.conferenceid=p.conferenceid 
                and j.journalid=p.journalid 
            order by authorid, paperid"""
        conjourcounts = data_io.run_query_db(query)
        this_table_feats = all_features[i]
        for ii,t in enumerate(conjourcounts): 
            this_table_feats[ii] +=t[2:]
           
     
def build_keywords_feature():    
    build_keywords_dict()
    global all_features 
    all_features = [] 
    for datatable in tables:
        print "building keyword features for "+datatable+"..."
        query = "select authorid, paperid, keyword from "+datatable+" left outer join paper on id=paperid order by authorid, paperid"
        apkeywords = data_io.run_query_db(query)
        features = []
        for aid, pid, keywords in apkeywords:
            #titles_set = get_title_set(int(aid))
            words = set()
            #keywords += title
            if(keywords!=None or keywords!=""):           
                w = re.split(regexPattern, keywords) #array of keywords
                for ww in w:
                    if(ww==""):
                        continue
                    words.add(ww)
            #now build feature for this author-paper example:  authorid, paperid, numofkeywordsofthispaper, numofkeywordsofauthor, numofcommonkeywords
            numofkeywordsofthispaper = len(words)
            numofkeywordsofauthor = len(author_keywords.get(aid))
            numofcommonkeywords = 0
            for pw in words:
                if pw in author_keywords.get(aid).keys():
                    numofcommonkeywords+=author_keywords.get(aid).get(pw)
        
#            print aid, " ", pid, " ", numofkeywordsofthispaper, numofkeywordsofauthor, numofcommonkeywords
            features.append((aid, pid, numofkeywordsofthispaper, numofkeywordsofauthor, numofcommonkeywords))
        all_features.append(features)
    
    

def build_additional_features():
    global all_features
    build_keywords_feature()
    build_confnJour_feature()
    build_titles_feature()
    print "saving additional features"
    with open("additional_features.obj", 'w') as dumpfile:
        cPickle.dump(all_features, dumpfile, protocol=cPickle.HIGHEST_PROTOCOL)


def get_additional_features():
    global all_features
    if(os.path.exists("additional_features.obj")):
        print "Loading additional feature..."
        with open("additional_features.obj", 'r') as loadfile:
            all_features = cPickle.load(loadfile)
    else:    
        build_additional_features()
        
    return all_features          
                
                 
if __name__=="__main__":
    get_additional_features()
    #build_titles_feature()
    #build_titles_feature()
    
    
    
    
    
    
    
    
    
    
    