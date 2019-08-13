"""
A pass through labeler
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

import os
import pickle
from image_recognition.abstract_labeler import AbstractLabeler


class IdentityLabeler(AbstractLabeler):
    def __init__(self, configObj, loader):
        self.loader = loader
        self.config = configObj
        self.database = configObj["database_engine"]
        self.database_name = configObj["database_name"]
        self.location = configObj["location"]
        self.data_folder_prefix = configObj["data_folder_prefix"]
        self.logger = configObj["logger"]

        super().__init__(configObj, loader)
        self.logger.debug("IdentityLabeler constructed")

    def run_labeler(self, filename="iden_labels_default.p", rerun=False):
        """
        loads labels from disk or uses labeler api to labeles images if
        there are no labels on disk, then saves the labels to disk in
        config.data_folder_prefix/filename
        """
        label_path = os.path.join(self.config["data_folder_prefix"], filename)
        if rerun or not os.path.exists(label_path):
            depths = self.loader.get_flood_depths()
            labels = self.get_labels(depths, self.dump_labels_to_disk)
        else:
            labels = self.load_labels_from_disk(filename)
        return labels

    def load_labels_from_disk(self, filename="iden_labels_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        return pickle.load(open(path, "rb"))

    def dump_labels_to_disk(self, labels, filename="iden_labels_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        pickle.dump(labels, open(path, "wb"))
        return

    def make_matrix(self, feat_vects):
        return super().make_matrix(feat_vects)

    def make_feature_vectors(self, inp, allowed):
        """
        Args:
            inp:
                Dictionary of (pkeys: int)
                where for each pkey represents the flood depth for pkey
            allowed:
                Dictionary of allowed word to the index in the feature vector
                example: allowed = {
                                    "Flood":0,
                                    "Flooding":1,
                                    "Water":2,
                                    "Puddle":3,
                                    "Person":4
                                    }
                would create feature vectors  where the zeroth
                feature is the confidence score of
                Flood in picture, 1st element is Flooding and so on
        Returns:
            Dictionary{ string Pkey: list{float}}
                where list is a vector defined by allowed
        """
        # dict of pkeys to feature vectors

        flood = self.config["flood_pkeys"]
        all_selected_pkeys = flood.union(self.config["no_flood_pkeys"])

        features = dict([(key, [0]*len(allowed.keys())) for key in inp.keys()
                        if key in all_selected_pkeys])
        for pkey in features.keys():
            all_selected_pkeys.remove(pkey)
            # fill in the label if it exists
            if pkey in inp:
                desc = "flood_depth"
                if desc in allowed:
                    features[pkey][allowed[desc]] =\
                            float(inp[pkey])
        # add in zero features that don't have flood heights
        zero_list = [0]*len(allowed.keys())
        for pkey in all_selected_pkeys:
            assert(pkey not in features)
            features[pkey] = zero_list

        self.features = features
        return features

    def get_labels(self, depths_df, hook=None):
        """
        Returns
        Args:
            depths_df(pd.DataFrame):
                pkeys to integer flood depths (in cm)
        Returns:
            Labels dict of (pkey, int)):
                for each pkey in depths_df, the integer result
                for that pkey
        """
        labels = depths_df.to_dict()["flood_depth"]
        if hook is not None:
            hook(labels)

        return labels

    def make_label_to_index(self, inp):
        """
        Args:
            inp dict of (pkeys: int)
                flood depth
        Returns:
            lab_to_index: dict(string: index)
                constant of {"flood_depth": 0}
            index_to_label: dict(index: string)
                constant of {0: "flood_depth"}
        """
        lab_to_index = {"flood_depth": 0}
        index_to_label = {0: "flood_depth"}
        return lab_to_index, index_to_label
