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

from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import os


class AbstractLabeler(ABC):
    def __init__(self, config, loader):
        self.config = config
        self.data_folder_prefix = config["data_folder_prefix"]
        self.loader = loader
        if not os.path.exists(self.data_folder_prefix):
            self.logger.debug(
                "data folder doesn't exist, creating path:",
                self.data_folder_prefix)
            os.makedirs(self.data_folder_prefix)

    @abstractmethod
    def load_labels_from_disk(self, filename='./abstract_labels.p'):
        pass

    @abstractmethod
    def dump_labels_to_disk(self, labels, filename='./abstract_labels.p'):
        pass

    @abstractmethod
    def make_feature_vectors(self, inp, allowed):
        pass

    @abstractmethod
    def get_labels(self, image_urls, hook=None):
        pass

    def run_labeler(self, filename):
        """
        loads labels from disk or uses labeler api to labeles images if
        there are no labels on disk, then saves the labels to disk in
        config.data_folder_prefix/filename
        """
        label_path = os.path.join(self.config["data_folder_prefix"], filename)
        if not os.path.exists(label_path):
            img_urls = self.loader.get_image_urls()
            labels = self.get_labels(img_urls, self.dump_labels_to_disk)
        else:
            labels = self.load_labels_from_disk(filename)
        return labels

    def make_matrix(self, feat_vects):
        """
        Adds true labels, turns into a numpy matrix
        Args:
            feat_vects: dict of pkey -> feat_vect
        Returns: numpy matricies
            Data_w_pkey
            Labels
        """
        mat = pd.DataFrame.from_dict(feat_vects, orient='index')
        mat["pkey"] = mat.index
        # make pkey the first column
        cols = mat.columns.to_list()[:-1]
        mat = mat[["pkey",  *cols]]
        # make true_flood the last column

        def add_true_data(inp_df):
            flood = self.config['flood_pkeys']
            for pkey in flood:
                inp_df.loc[inp_df['pkey'] == pkey, 'true_flood'] = 1
            inp_df.loc[inp_df['true_flood'] != 1, 'true_flood'] = -1
        add_true_data(mat)
        mat = mat.to_numpy().T
        # now first row is pkey, w/ the corresponding col vector and the
        # true label at the end
        true_labels = mat[-1, :]
        true_labels = np.reshape(true_labels, (1, true_labels.shape[0]))

        data = mat[:-1, :]
        return data, true_labels
