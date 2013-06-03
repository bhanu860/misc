'''
Created on May 8, 2013

@author: bhanu
'''
from FeatureExtractor import FeatureExtractor
import os

if __name__ == '__main__':
    
    fe = FeatureExtractor()
    fe.load_dup_dict()
#    fe = FeatureExtractor()
#    if(os.path.exists("instanceList.obj")): 
#        fe.load_data()
#    else:
#        fe.extractFeatures()
#        
#    fe.rolf()