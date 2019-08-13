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

import pickle
import os

import pandas as pd
import numpy as np


import boto3 as boto3
from botocore.exceptions import ClientError

import requests
import io

from PIL import Image
from sqlalchemy import create_engine

from image_recognition.abstract_labeler import AbstractLabeler

global engine
DATABASE = "cognicity"
engine = create_engine(
    "postgresql://postgres:postgres@localhost:5432/"
    + DATABASE)

CH_DATABASE = "riskmap"
CH_ENGINE = create_engine(
    "postgresql://postgres:postgres@localhost:5432/"
    + CH_DATABASE)


class AwsLabeler(AbstractLabeler):
    def __init__(self, configObj, loader):
        self.config = configObj
        self.database = configObj["database_engine"]
        self.database_name = configObj["database_name"]
        self.location = configObj["location"]
        self.data_folder_prefix = configObj["data_folder_prefix"]
        self.logger = configObj["logger"]

        super().__init__(configObj, loader)
        self.logger.debug("AwsLabeler constructed")

    def load_labels_from_disk(self, filename="aws_labels_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        self.logger.debug("logging from: " + str(path))
        return pickle.load(open(path, "rb"))

    def dump_labels_to_disk(self, labels, filename="aws_labels_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        pickle.dump(labels, open(path, "wb"))
        return

    def run_labeler(self, filename="aws_labels_default.p"):
        return super().run_labeler(filename)

    def make_matrix(self, feat_vects):
        return super().make_matrix(feat_vects)
        pass

    def make_label_to_index(self, inp):
        """
        Args:
            inp:
                Dictionary of pkeys -> httpResponse from aws
        Returns:
            lab_to_index: dict(string: index)
            index_to_label: dict(index: string)
        """
        all_labels = set()
        for key, each in inp.items():
            for lab in each['Labels']:
                if lab['Name'] not in all_labels:
                    all_labels.add(lab['Name'])
        # allowed has label: index pairs for ex: 'Flood':0,
        # so 'Flood' is the first
        # in the feature vector
        # all labels from AWS
        lab_to_index = dict([(current_label, index) for index, current_label
                            in enumerate(list(all_labels))])
        index_to_label = dict([(index, current_label) for index, current_label
                               in enumerate(list(all_labels))])
        return lab_to_index, index_to_label

    def make_feature_vectors(self, inp, allowed):
        """
        Args:
            inp:
                Dictionary of pkeys-> httpResponses from aws.
            allowed:
                Dictionary of allowed word to the index in the feature vector
                example: allowed = {"Flood":0, "Flooding":1, "Water":2,
                                    "Puddle":3, "Person":4}
                would create feature vectors  where the zeroth feature is
                the confidence score of
                Flood in picture, 1st element is Flooding and so on
        Returns:
            Dictionary{ string Pkey: list{float}}:  where list
                                is a vector defined by allowed
        """
        flood = self.config["flood_pkeys"]
        remaining_pkeys = flood.union(self.config["no_flood_pkeys"])

        # dict of pkeys to feature vectors
        # feat vects are intialized to zero
        features = dict([(key, [0]*len(allowed.keys())) for key in inp.keys()
                        if key in remaining_pkeys])
        for pkey in features.keys():
            remaining_pkeys.remove(pkey)
            # if we labeled this one then add it, otherwise let it be zero
            if pkey in inp:
                from_aws = inp[pkey]["Labels"]
                for tag in from_aws:
                    if tag["Name"] in allowed:
                        confidence = float(tag["Confidence"])
                        features[pkey][allowed[tag["Name"]]] = confidence

        # add in zero features that don't have images
        zero_list = [0]*len(allowed.keys())
        for pkey in remaining_pkeys:
            assert(pkey not in features)
            features[pkey] = zero_list

        self.features = features
        return features

    def get_labels(self, image_urls, hook=None):
        labels = dict()
        for pkey, img_url in image_urls.items():
            try:
                r = requests.get(img_url)
                if r.status_code == 200:
                    img_bytes = io.BytesIO(r.content).getvalue()
                    self.logger.debug('requesting: ' + img_url)
                    client = boto3.client("rekognition",
                                          region_name='ap-south-1')
                    labels[pkey] = client.detect_labels(
                        Image={
                            "Bytes": img_bytes
                            },
                        MinConfidence=.10)
                    if hook is not None:
                        hook(labels)
                else:
                    self.logger.info(
                                "could not fetch image " + str(img_url))
            except ClientError as e:
                self.logger.error("AWS client error ", e)
        return labels


def get_image_urls():
    '''
    returns dictionary of {pkey: image_url}
    for all rows in db that have an image url
    '''
    global engine
    rows = pd.read_sql_query('''
    SELECT pkey, image_url FROM cognicity.all_reports
        WHERE image_url IS NOT null
        ORDER BY created_at
    ''', con=engine, index_col="pkey")
    return rows.to_dict()['image_url']

def get_image_urls_chennai():
    '''
    returns dictionary of {pkey: image_url} for all rows in db that have an image url
    '''
    rows = pd.read_sql_query('''
    SELECT pkey, image_url FROM riskmap.all_reports
        WHERE image_url IS NOT null
        ORDER BY created_at
    ''', con=CH_ENGINE, index_col="pkey")
    return rows.to_dict()['image_url']

def get_df():
    global engine
    rows = pd.read_sql_query('''
    SELECT pkey, created_at, text, image_url FROM cognicity.all_reports 
        WHERE (image_url IS NOT null) AND ( text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%') 
        ORDER BY created_at
    ''', con=engine, index_col="pkey")
    
    print(rows.columns.values)
    return rows

def fetch_images(location='id'):
    image_urls = get_image_urls()
    for key in image_urls.keys():
        each = image_urls[key]
        print("url is: " + each)
        r = requests.get(each, stream=True)
        if r.status_code == 200:
            try:
                im = Image.open(r.raw)
                if location == 'id':
                    im.save("./img/"+ str(key) + ".jpeg", "JPEG")
                else:
                    im.save("./img_ch/"+ str(key) + ".jpeg", "JPEG")
            except:
                print("ERROR FETCHING", each)
        else: 
            print("ERROR COULD NOT READ URL: " + each)

def modify_df(df):
    working_rows = []
    for index, row in df.iterrows():
        url = row["image_url"]

        print("url is: " + url)
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            img = Image.open(r.raw)
            #labels = get_labels_aws(img.tobytes())
            labels = get_labels_aws_from_bucket(url)
            print(labels)
            working_rows.append(row)
        else: 
            print("ERROR COULD NOT READ URL: " + url)
            print(row)

def get_labels_aws(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            img_bytes = io.BytesIO(r.content).getvalue()
            print('requesting: ' + url)
            client = boto3.client("rekognition", region_name='ap-south-1')
            return client.detect_labels(
                Image={
                    "Bytes": img_bytes
                    },
                MinConfidence=.10)
        else:
            return dict()
    except:
        return dict()

def get_labels_aws_from_bucket(url):
    name = url.split("riskmap.in/")[-1] # last part of url
    print("name is " + name)
    client = boto3.client("rekognition", region_name="ap-south-1")
    return client.detect_labels(
            Image={
                "S3Object" : {
                    "Bucket": "images.riskmap.in",
                    "Name": "originals/"+ name
                    }
                })
    

def filter_labels(wanted, given):
    """
    Args:
        wanted: The labels to pick out of given if they exist

        given: The dictionary returned from detect_labels aws endpoint

    Returns:
        Dictionary{string: float} of "label": confidence score tuples
    """
    res = dict()
    for each in wanted:
        if each in given:
            res[each] = given[each]
    return res

def dump_labels_to_disk(filename="./labels.p", location="id"):
    if location == "id":
        img_urls = get_image_urls()
    else:
        img_urls = get_image_urls_chennai()
    pkey_to_labels = dict()
    for pkey in img_urls.keys():
        print(pkey)
        labels = get_labels_aws(img_urls[pkey])
        pkey_to_labels[pkey] = labels
        # dict to json and then put in the database
        # ALTER TABLE ADD COLUMN "labels"  of type json
        # ALTER TABLE ADD COLUMN "feature_vector"
        pickle.dump(pkey_to_labels, open(filename, "wb"))

def read_labels_from_disk(filename="./labels.p"):
    return pickle.load(open(filename, "rb"))

def clean_if_dirty(data):
    res = {}
    toDelete = []
    for key in data.keys():
        if not data[key]:
            print("ERROR: MISSING DATA FOR PKEY:", key)
            toDelete.append(key)
    for each in toDelete:
        del data[each]
    return data

def make_feature_vectors(inp, allowed):
    """ 
    Args:
        inp:
            Dictionary of pkeys-> httpResponses from aws.
        allowed: 
            Dictionary of allowed word to the index in the feature vector

            example: allowed = {"Flood":0, "Flooding":1, "Water":2, "Puddle":3, "Person":4}
            would create feature vectors  where the zeroth feature is the confidence score of 
            Flood in picture, 1st element is Flooding and so on
    
    
    Returns:
        Dictionary{ string Pkey: list{float}}  where list is a vector defined by allowed

    """
    # dict of pkeys to feature vectors
    features = dict([ (key, [0]*len(allowed.keys())) for key in inp.keys()] )
    for pkey in inp.keys():
        # print(inp[pkey])
        # print("key: ", pkey)
        from_aws = inp[pkey]["Labels"]
        for tag in from_aws:
            if tag["Name"] in allowed:
                features[pkey][allowed[tag["Name"]]] = float(tag["Confidence"])
                # print(pkey)
                # print(features[pkey])
        #print("pkey", pkey)
        #print(features)
    return features

def make_matrix_rep(featureDict, lenFeatVect):

    # looks like: 
    # pkey0    | pkey1 .. 
    # featvect | featV1
    out = np.zeros((lenFeatVect + 1, len(featureDict.keys())))
    for i, pkey in enumerate(sorted(featureDict)):
        l = featureDict[pkey].copy()# shallow copy because they're builtins
        l.insert(0, pkey)
        out[:,i] = np.array(l)
    return out

def make_matrix_rep_zeros(featureDict, lenFeatVect):

    # looks like: 
    # pkey0    | pkey1 .. 
    # featvect | featV1

    # if pkey=i doesn't exist then creates a zero column
    out = np.zeros((lenFeatVect, max(featureDict.keys())))
    for i in range(max(featureDict.keys())):
        if i in featureDict:
            print(i)
            l = featureDict[i].copy() # shallow copy because they're builtins
            print(l)
            out[:,i] = np.array(l)
    return out


def make_labels_rep(featureDict, location='id'):
    # if pkey exists in feature dict, figures out if flooding
    # else zero

    #start_known_flood = "'2017-02-20 00:00:35.630000-05:00'"
    #end_known_flood = "'2017-02-23 00:00:35.630000-05:00'"
    if location == 'id':
        start_known_flood = "2017-02-20 00:00:35.630000-05:00"
        end_known_flood =   "2017-02-23 00:00:35.630000-05:00"
        global engine
        knownFlood = pd.read_sql_query('''
            SELECT pkey from cognicity.all_reports
            WHERE created_at > '2017-02-20 00:00:35.630000-05:00'
            AND created_at < '2017-02-23 00:00:35.630000-05:00'
        ''', con=engine, params={"start_known_flood": start_known_flood, "end_known_flood":end_known_flood})

    else:
        start_known_flood = "'2017-11-01 00:00:35.630000-04:00'"
        end_known_flood = "'2017-11-07 00:00:35.630000-04:00'"

        knownFlood = pd.read_sql_query('''
            SELECT pkey from riskmap.all_reports
            WHERE created_at > %(start_known_flood)s::timestamptz
            AND created_at < %(end_known_flood)s::timestamptz
        ''', con=CH_ENGINE, params={"start_known_flood": start_known_flood, "end_known_flood":end_known_flood})

    knownFloodSet = set(knownFlood['pkey'])
    print(knownFloodSet)

    out = np.zeros((2, len(featureDict.keys())))
    for i, pkey in enumerate(sorted(featureDict)):
        # look up if this pkey is a flood event
        if pkey in knownFloodSet:
            out[0, i] = pkey
            out[1, i] = 1
        else:
            # no known flooding
            out[0, i] = pkey
            out[1, i] = -1
    return out


def make_labels_rep_zeros(featureDict):
    # if pkey exists in feature dict, figures out if flooding
    # else zero

    # start_known_flood = "'2017-02-20 00:00:35.630000-05:00'"
    # end_known_flood = "'2017-02-23 00:00:35.630000-05:00'"

    start_known_flood = "2017-02-20 00:00:35.630000-05:00"
    end_known_flood = "2017-02-23 00:00:35.630000-05:00"
    # TODO Make the dates programatic
    global engine
    knownFlood = pd.read_sql_query('''
        SELECT pkey from cognicity.all_reports
        WHERE created_at > '2017-02-20 00:00:35.630000-05:00'
        AND created_at < '2017-02-23 00:00:35.630000-05:00'
    ''', con=engine, params={"start_known_flood": start_known_flood, "end_known_flood":end_known_flood})

    knownFloodSet = set(knownFlood['pkey'])
    print(knownFloodSet)

    out = np.zeros((1, max(featureDict.keys())))
    for i in range(max(featureDict.keys())):
        if i in featureDict:
            pkey = i
            # look up if this pkey is a flood event
            if pkey in knownFloodSet:
                out[0, i] = 1
            else:
                # no known flooding
                out[0, i] = -1
        else:
            # zero 
            out[0, i]
    return out

def dump_labels_to_disk_chennai(filename='./min_confidence_chennai_labels.p'):
    dump_labels_to_disk(filename, 'ch')
    return

if __name__ == "__main__":
    dump_labels_to_disk_chennai()
    # dump_labels_to_disk("./min_confidence.p")
    labes = read_labels_from_disk()
