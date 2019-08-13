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
        """
        Args:
            image_urls(pd.DataFrame)
        Returns:
            labeler specific encoding of
            pkeys -> labels
        """
        # not implemented, should never be called
        assert(False)
        pass

    def run_labeler(self, filename, rerun=False):
        """
        loads labels from disk or uses labeler api to labeles images if
        there are no labels on disk, then saves the labels to disk in
        config.data_folder_prefix/filename
        """
        label_path = os.path.join(self.config["data_folder_prefix"], filename)
        if rerun or not os.path.exists(label_path):
            img_urls = self.loader.get_image_urls()
            labels = self.get_labels(img_urls, self.dump_labels_to_disk)
        else:
            labels = self.load_labels_from_disk(filename)
        return labels

    def __add_true_data(self, inp_df):
        flood = self.config['flood_pkeys']
        for pkey in flood:
            inp_df.loc[inp_df['pkey'] == pkey, 'true_flood'] = 1
        inp_df.loc[inp_df['true_flood'] != 1, 'true_flood'] = -1

    def make_matrix(self, feat_vects):
        """
        Adds true labels, turns into a numpy matrix
        Args:
            feat_vects: dict of pkey -> feat_vect
        Returns:
            data: numpy.ndarray
                matrix with columns that are feat vects
                columns are sorted by pkey
            true_labels
                +1 if flood, -1 otherwise
        """
        mat = pd.DataFrame.from_dict(feat_vects, orient='index')
        mat = mat.sort_index()
        mat["pkey"] = mat.index
        # make pkey the first column
        cols = mat.columns.to_list()[:-1]
        mat = mat[["pkey",  *cols]]
        # make true_flood the last column

        self.__add_true_data(mat)
        mat = mat.to_numpy().T
        # now first row is pkey, w/ the corresponding col vector and the
        # true label at the end
        true_labels = mat[-1, :]
        true_labels = np.reshape(true_labels, (1, true_labels.shape[0]))

        data = mat[:-1, :]
        return data, true_labels

    def to_embedding_projector(self, index_to_label, feat_dict, prefix):
        """
        Creates files for visualization in tensorflow projector.
        writes two files, metadata.tsv and vectors.tsv
        into the data folder specified in config.
        Args:
            index_to_label: dict (int:string)
            feat_dict: dict(pkey: list(float))
            prefix(String) : prefix for filename
        """

        img_urls = self.loader.get_image_urls()
        img_df = pd.DataFrame.from_dict(img_urls,
                                        orient="index",
                                        columns=["url"])
        img_df.index.rename("pkey")
        img_df.rename_axis("pkey")

        feature_mat = pd.DataFrame.from_dict(feat_dict, orient='index').rename(
                columns=index_to_label)
        feature_mat.index = feature_mat.index.rename("pkey")

        meta_df = pd.concat([img_df, feature_mat], axis=1, join="inner")
        meta_df.rename_axis(index="pkey", axis="index", inplace=True)

        df = feature_mat
        vectors_tab = df.to_csv(sep='\t', header=False, index=False)
        path = os.path.join(self.data_folder_prefix, prefix+"_vectors.tsv")
        v = open(path, 'w')
        v.write(vectors_tab)
        v.close()

        # add in the url
        # meta_df = pd.concat([img_df, feature_mat], axis=1, join="inner")
        meta_df = pd.concat([img_df, feature_mat], axis=1, join="inner")
        meta_df.rename_axis(index="pkey", axis="index", inplace=True)
        meta_df["pkey"] = meta_df.index
        self.__add_true_data(meta_df)
        metadata = meta_df.to_csv(sep='\t', index=False)

        meta_path = os.path.join(self.data_folder_prefix,
                                 prefix + "_metadata.tsv")
        f = open(meta_path, 'w')
        f.write(metadata)
        f.close()
