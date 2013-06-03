'''
Created on May 4, 2013

@author: bhanu
'''
from sets import Set


class Instance(object):
    '''
    Instace represents an author
    '''
    id = None
    name = None
    co_authors = None 
    keywords = None
    confNJourIds = None
    affiliations = None
    input = None
    output= None
    hasNoPapers = False

    def __init__(self):
        '''
        Constructor
        '''
        self.name = Set()
        self.co_authors = Set()
        self.keywords = Set()
        self.confNJourIds = Set()
        self.affiliations = Set()
