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

import os
import pickle
from image_recognition.abstract_labeler import AbstractLabeler

from google.api_core.exceptions import GoogleAPICallError
from google.cloud.vision_v1 import ImageAnnotatorClient


client = ImageAnnotatorClient()


class GoogleLabeler(AbstractLabeler):
    def __init__(self, configObj, loader):
        self.loader = loader
        self.config = configObj
        self.database = configObj["database_engine"]
        self.database_name = configObj["database_name"]
        self.location = configObj["location"]
        self.data_folder_prefix = configObj["data_folder_prefix"]
        self.logger = configObj["logger"]

        super().__init__(configObj, loader)
        self.logger.debug("GoogleLabeler constructed")

    def run_labeler(self, filename="goog_labels_default.p"):
        return super().run_labeler(filename)

    def load_labels_from_disk(self, filename="goog_labels_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        return pickle.load(open(path, "rb"))

    def dump_labels_to_disk(self, labels, filename="goog_labels_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        pickle.dump(labels, open(path, "wb"))
        return

    def make_matrix(self, feat_vects):
        return super().make_matrix(feat_vects)

    def make_feature_vectors(self, inp, allowed):
        """
        Args:
            inp:
                Dictionary of pkeys-> AnnotateImageResponse from google cloud.
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
                labels = inp[pkey].label_annotations
                for entityObject in labels:
                    desc = entityObject.description
                    if desc in allowed:
                        features[pkey][allowed[desc]] =\
                                        float(entityObject.score)
        # add in zero features that don't have images
        zero_list = [0]*len(allowed.keys())
        for pkey in all_selected_pkeys:
            assert(pkey not in features)
            features[pkey] = zero_list

        self.features = features
        return features

    def get_labels(self, image_urls, hook=None):
        """
        Gives the label response from Google
        Args:
            image_urls(pd.DataFrame): pkeys to urls
        Returns:
            Labels(dict(pkey, googleAnnotateRequest)):
                for each pkey in image_urls, the googleAnnotateRequest result
                for that image
        """
        labels = dict()

        for pkey, img_name in image_urls.items():
            try:
                request = {
                        "image": {
                            "source": {"image_uri": img_name},
                            },
                        "features": [
                            {
                                "type": "LABEL_DETECTION",
                                "max_results": 100
                                }
                            ]
                        }
                response = client.annotate_image(request)
                labels[pkey] = response

                if hook is not None:
                    self.logger.debug("labeled pkey " + str(pkey))
                    self.logger.debug("img url " + str(img_name))
                    hook(labels)
            except GoogleAPICallError:
                self.logger.debug("ERROR LABELING PKEY: " + str(pkey))
                self.logger.debug("WITH IMG URL: " + str(img_name))
        return labels

    def make_label_to_index(self, inp):
        """
        Args:
            inp:
                Dictionary of pkeys -> labelAnnotate resp from google
        Returns:
            lab_to_index: dict(string: index)
            index_to_label: dict(index: string)
        """
        all_labels = set()
        for key, each in inp.items():
            for lab in each.label_annotations:
                if lab.description not in all_labels:
                    all_labels.add(lab.description)

        lab_to_index = dict([(current_label, index) for index, current_label in
                             enumerate(list(all_labels))])
        index_to_label = dict([(index, current_label) for index, current_label
                               in enumerate(list(all_labels))])
        return lab_to_index, index_to_label


# if __name__ == "__main__":
#     LOGGER = logging.getLogger()
#     LOGGER.setLevel(logging.DEBUG)
#     formatter = logging.Formatter(
#                 "%(asctime)s - %(levelname)s - %(message)s")
#     TEST_LOG_FILENAME = "test_log_filename.log"
#     fh = logging.FileHandler(TEST_LOG_FILENAME)
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(formatter)
#     LOGGER.addHandler(fh)
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(formatter)
#     LOGGER.addHandler(ch)
#     DATABASE = "cognicity"
#     ID_ENGINE = create_engine(
#                 "postgresql://postgres:postgres@localhost:5432/"
#                 + DATABASE)
#     IMG_FOLDER_PREFIX = "100_labels_goog"
#     configObj = {
#             "database_engine": ID_ENGINE,
#             "database_name": "cognicity",
#             "location": "id",
#             "img_folder_prefix": IMG_FOLDER_PREFIX,
#             "logger": LOGGER}
#
#     loader = CognicityImageLoader(configObj)
#     img_urls = loader.get_image_urls()
#
#     config = {}
#     labeler = GoogleLabeler(config)
#     labels = labeler.get_labels(img_urls, hook=labeler.dump_labels_to_disk)
#
#     # labels = labeler.load_labels_from_disk()
#
#     all_labels = set()
#     for key, each in labels.items():
#         for lab in each.label_annotations:
#             if lab.description not in all_labels:
#                 all_labels.add(lab.description)
#
#     ALL_LABELS = dict([(current_label, index)
#                       for index, current_label
#                       in enumerate(list(all_labels))])
#     print(len(ALL_LABELS))
#     print(ALL_LABELS)
#
#     feat = labeler.make_feature_vectors(labels, ALL_LABELS)
