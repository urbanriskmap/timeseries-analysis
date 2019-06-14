"""
Reads images from database and fetches labels from aws
Abraham Quintero, MIT 2018
v0.0.1
"""

__author__ = "Abraham Quintero"
__created__ = "2019-2-9"
__version__ = "0.0.1"
__copyright__ = "Copyright 2019 MIT Urban Risk Lab"
__license__ = "MIT"
__email__ = "abrahamq@mit.edu"
__status__ = "Development"
__url__ = "https://github.com/urbanriskmap/timeseries-analysis"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import requests
import base64
import io
import pickle

from PIL import Image
from matplotlib.animation import FuncAnimation

import datetime
from sqlalchemy import create_engine

import common as util

from google.cloud.vision_v1 import ImageAnnotatorClient
client = ImageAnnotatorClient()

def load_labels_from_disk(filename='./goog_labels_chennai.p'):
    return pickle.load(open(filename, 'rb'))

def dump_labels_to_disk(labels, filename='./goog_labels_chennai.p'):
    pickle.dump(labels, open(filename, 'wb'))
    return

def make_feature_vectors(inp, allowed):
    ''' 
    Args:
        inp: 
            Dictionary of pkeys-> AnnotateImageResponse from google cloud.
        allowed: 
            Dictionary of allowed word to the index in the feature vector 

            example: allowed = {'Flood':0, 'Flooding':1, 'Water':2, 'Puddle':3, 'Person':4}
            would create feature vectors  where the zeroth feature is the confidence score of 
            Flood in picture, 1st element is Flooding and so on
    
    
    Returns:
        Dictionary{ string Pkey: list{float}}  where list is a vector defined by allowed

    '''
    # dict of pkeys to feature vectors
    features = dict([ (key, [0]*len(allowed.keys())) for key in inp.keys()] )
    for pkey in inp.keys():
        # print(inp[pkey])
        # print('key: ', pkey)
        labels = inp[pkey].label_annotations
        for entityObject in labels:
            desc = entityObject.description
            if desc in allowed:
                features[pkey][allowed[desc]] = float(entityObject.score)
        #print('pkey', pkey)
        #print(features)
    return features


def get_labels(image_urls, hook=None):
    labels = dict()

    for pkey, img_name in image_urls.items():
        try:
            request = {
                    'image': {
                        'source': {'image_uri': img_name},
                        },
                    }
            response = client.annotate_image(request)
            labels[pkey] = response
            print(response)
    
            if hook is not None:
                print('labeled pkey ', pkey)
                print('img url ', img_name)
                hook(labels)
        except:
            print('ERROR LABELING PKEY: ', pkey)
            print('WITH IMG URL: ', img_name)
    return labels

if __name__ == "__main__":

    #test_img_url = { 1: 'https://images.riskmap.in/Sym-2DodW.jpg'}
    #labels = get_labels(test_img_url, hook=dump_labels_to_disk)

    #urls = util.get_image_urls()
    ## pkey to url
    #labels = get_labels(urls, hook=dump_labels_to_disk)
    labels = load_labels_from_disk()

    all_labels = set()
    for key, each in labels.items():
        for lab in each.label_annotations:
            if lab.description not in all_labels:
                all_labels.add(lab.description)

    ALL_LABELS = dict([ (current_label, index) for index, current_label in enumerate(list(all_labels))]) 
    print(len(ALL_LABELS))
    print(ALL_LABELS)

    feat = make_feature_vectors(labels, ALL_LABELS)
    print()


