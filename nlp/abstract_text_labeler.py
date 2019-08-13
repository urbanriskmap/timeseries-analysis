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

from abc import abstractmethod
import pickle

from image_recognition.abstract_labeler import AbstractLabeler


class AbstractTextLabeler(AbstractLabeler):
    def __init__(self, config, loader):
        self.config = config
        self.loader = loader
        super().__init__(config, loader)

    @abstractmethod
    def load_labels_from_disk(self, filename="./abstract_labels.p"):
        return pickle.load(open(filename, "rb"))

    @abstractmethod
    def dump_labels_to_disk(self, labels, filename="./abstract_labels.p"):
        pickle.dump(labels, open(filename, "wb"))
        return

    def prepare_text(self, inp):
        """
        Args:
            inp: (str)
                input sentence
        Returns:
            Tokenized: (list(str))
                a list of strings where each string is a different token
        """
        # TODO replace this with regex?
        import string
        exclude = set(string.punctuation)
        s = "".join(ch for ch in inp if ch not in exclude)
        tokenized = s.lower().replace("\n", " ").split()
        return tokenized

    @abstractmethod
    def make_feature_matrix(self, inp, allowed):
        pass

    @abstractmethod
    def get_labels(self, texts, hook=None):
        pass

    def __add_true_data(self, inp_df):
        """
        Modifies inp_df to include a column "true_flood"
        that is 1 if there was flooding, -1 if not
        Args:
            inp_df: (pd.DataFrame)
                with an index column "pkeys"
        """
        flood = self.config['flood_pkeys']
        for pkey in flood:
            inp_df.loc[inp_df['pkey'] == pkey, 'true_flood'] = 1
        inp_df.loc[inp_df['true_flood'] != 1, 'true_flood'] = -1
