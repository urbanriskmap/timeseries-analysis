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


class AbstractLabeler(ABC):
    def __init__(self, config):
        self.config = config

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
