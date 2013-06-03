import csv
import json
import os
import pickle
from scipy import io
import numpy as np


def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def save_model(model, prefix=""):
    out_path = get_paths()["model_path"]
    head, tail = os.path.split(out_path)
    new_path = os.path.join(head, prefix+tail)
    pickle.dump(model, open(new_path, "w"))
    

def load_model(prefix=""):
    in_path = get_paths()["model_path"]
    head, tail = os.path.split(in_path)
    new_path = os.path.join(head, prefix+tail)
    return pickle.load(open(new_path))

def write_submission(predictions, prefix=""):
    pass


def get_classid(train_filename, classidDict):
    #train_filename Eg: cepst_train_aegithalos_caudatus.mat
    #classidFilename Eg: train_aegithalos_caudatus.wav
    prefix = "cepst_"
    tf, _ = train_filename.split(".") #remove extension
    tff = tf[len(prefix):]
    key = tff+".wav"
    return classidDict[key]
    


def convert_to_theano(mats, targetFiles):
    key = "cepstra"    
    species_numbers = get_paths()["species_numbers"]
    #mats : list of dict objects
    #targetFiles: list  of file names
    #build target dictionary
    targetDict = dict()
    with open(species_numbers, "r") as spn:
        reader = csv.reader(spn)
        next(reader)
        for row in reader:
            targetDict[row[1]] = int(row[0])
        
    mat = (mats[0])[key]
    features = np.ndarray(shape=(mat.shape[0]*len(mats), mat.shape[1]))
    target = np.ndarray(shape=mat.shape[0]*len(mats), dtype='int32')
    for i, mat in enumerate(mats):
        temp = mat[key]
        features[i:i+temp.shape[0],: ] = temp#np.concatenate(features, temp)
        target[i:i+temp.shape[0]] = get_classid(targetFiles[i], targetDict)
    return features, target


def get_features_mat():
    key = "cepstra"
    
    data_path = get_paths()["data_dir"]
    #read  and append all the .mat files 
    fileNames = os.listdir(data_path)
    targetFiles = []
    mats = []
    for f in fileNames:
        ff = str(f)
        if(ff.endswith(".mat")):
            with open(data_path+ff, "r") as matfile: 
               mat = io.loadmat(matfile)
               mats.append(mat)
               targetFiles.append(ff)
    return convert_to_theano(mats, targetFiles) #i.e. to numpy ndarrays
               
                
    
    print mats
    
def get_features_csv():
    prefix = "cepst_"
    
    data_path = get_paths()["data_dir"]
    species_numbers = get_paths()["species_numbers"]
    #build target dictionary
    targetDict = dict()
    with open(species_numbers, "r") as spn:
        reader = csv.reader(spn)
        next(reader)
        for row in reader:
            targetDict[row[0]] = row[1]
    fileNames = os.listdir(data_path)
    features = []; target = []
    for f in fileNames:
        ff = str(f)
        if(ff.endswith(".csv")):
            with open(data_path+ff, "r") as csvfile: 
                reader = csv.reader(csvfile)
                for row in reader:
                    floats = [float(x) for x in row]
                    features.append(floats) 
    
   

