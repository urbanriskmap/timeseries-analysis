import pickle
import os
import numpy as np
from abc import ABC


class AbstractLearner(ABC):

    def __init__(self, config, loader, labeler):
        """
        Args:
            config: dict of (str: obj)
                defined by chennai_config or jakarta_config
            loader: class that implements abstract_loader
            labeler: class that implements abstract_labeler
        """
        self.config = config
        self.logger = config["logger"]
        self.data_folder_prefix = config["data_folder_prefix"]
        self.loader = loader(config)
        self.labeler = labeler(self.config, self.loader)

    def load_model_from_disk(self, filename="perceptron_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        self.logger.debug("logging from: " + str(path))
        return pickle.load(open(path, "rb"))

    def dump_model_to_disk(self, sep, filename="perceptron_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        pickle.dump(sep, open(path, "wb"))
        return

    def train(self, params, validation_keys):
        """
        Args:
            params dict of (str: obj)
                passed onto the perceptron function
                "T": number of iterations
                "print": True/False whether to print progress
            validation_keys: set(int)
                pkeys that should not be used for training
        Returns:
            (th, th0) tuple of  numpy.ndarray:
                linear model and an offset
        """
        pass

    def run_learner(self,
                    filename,
                    rerun=False,
                    validation_keys=dict(),
                    params={"T": 1000, "print": True}):
        path = os.path.join(self.data_folder_prefix, filename)
        if rerun or not os.path.exists(path):
            model = self.train(params, validation_keys=validation_keys)
            self.dump_model_to_disk(model, filename)
        else:
            model = self.load_model_from_disk(filename)
        return model

    def predict(self, datapoint):
        """
        Args:
            datapoint (numpy.ndarray)
                Must have the same length as th
        Returns:
            signed distance from model
        """
        pass

    def cross_validate_model(self,
                             data,
                             labels,
                             k=5,
                             params={
                                 "T": 1000,
                                 "print": False
                                 }
                             ):
        """
        Rnadomly shuffles data and labels into
        k many groups. Trains on k-1 and then test
        on left out group.
        Args:
            data
            labels


        """
        full = np.vstack((data, labels)).copy()
        np.random.shuffle(full.T)
        data = full[:-1, :]
        labels = full[-1:, :]
        # axis=1 means columns, split into groups of cols
        s_data = np.array_split(data, k, axis=1)
        s_labels = np.array_split(labels, k, axis=1)
        score_sum = 0
        for i in range(k):
            data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
            labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:],
                                          axis=1)
            data_test = np.array(s_data[i])
            labels_test = np.array(s_labels[i])
            self._train(data_train, labels_train, params=params)
            # number correctly predicted in test set
            this_score = self._score(data_test, labels_test)
            score_sum += this_score
