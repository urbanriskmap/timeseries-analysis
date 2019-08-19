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

import boto3 as boto3
from botocore.exceptions import ClientError

import requests
import io

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

    def make_feature_vectors(self,
                             inp,
                             allowed,
                             include_zero_vects=True):
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
        if include_zero_vects:
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
