from learners.abstract_learner import AbstractLearner

from sklearn import svm
import numpy as np


class SvmLearner(AbstractLearner):
    """
    passes scalar raw features

    """
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
        super().__init__(config, loader, labeler)

    def load_model_from_disk(self, filename="identity_default.p"):
        return super().load_model_from_disk(filename)

    def dump_model_to_disk(self, sep, filename="identity_default.p"):
        super().dump_model_to_disk(sep, filename)
        return

    def train(self, params, validation_keys):
        """
        Args:
            params dict of (str: obj)
                passed onto the ml.perceptron function
                "T": number of iterations
                "print": True/False whether to print progress
            validation_keys: set(int)
                pkeys that should not be used for training
        Returns:
            (th, th0) tuple of  numpy.ndarray:
                linear model and an offset
        """
        labels = self.labeler.run_labeler()
        lab_to_index, index_to_lab = self.labeler.make_label_to_index(labels)

        self.lab_to_index = lab_to_index
        self.index_to_lab = index_to_lab
        feat = self.labeler.make_feature_vectors(labels, lab_to_index,
                                                 include_zero_vects=False)
        training_feat = dict()
        validation_feat = dict()
        for key, val in feat.items():
            if key not in validation_keys:
                training_feat[key] = val
            else:
                validation_feat[key] = val
        # train
        t_data_w_pkey, t_labels = self.labeler.make_matrix(training_feat)
        self.t_data_w_pkey = t_data_w_pkey
        self.t_labels = t_labels
        t_data = t_data_w_pkey[1:, :]

        # found best hyper params by exhaustive grid search in svm_test.ipynb
        self.clf = svm.SVC(gamma="scale", kernel="rbf", degree=3)
        # sklearn expects rows to be data points, we've gone with columns
        self.clf.fit(t_data.T, self.t_labels[0, :])

        # validation
        val_data_w_pkey, val_labels = self.labeler.make_matrix(validation_feat)
        self.val_data_w_pkey = val_data_w_pkey
        self.val_labels = val_labels
        val_data = val_data_w_pkey[1:, :]

        pred = self.clf.predict(val_data.T)

        correct = np.sum(val_labels == pred)

        total = val_data.shape[1]
        percent_correct = correct/total
        self.logger.info("Num Correct " + str(correct) +
                         " Out of " + str(total))
        self.logger.info("Val score: " + str(percent_correct))

        # get the signed distance for every train data point
        # with the pkey as the first row
        # self.t_sd = np.vstack((self.t_data_w_pkey[0, :],
        #                       self.clf.decision_function(t_data.T)))
        # # for every validation data point
        # self.val_sd = np.vstack((self.val_data_w_pkey[0, :],
        #                         self.clf.decision_function(val_data.T)))

        self.t_sd = self.clf.decision_function(t_data.T)
        # for every validation data point
        self.val_sd = self.clf.decision_function(val_data.T)

        return val_data, val_labels

    def predict(self, datapoint):
        """
        Args:
            datapoint (numpy.ndarray)
                Must have the same length as th
        Returns:
            signed distance from model
        """
        if not self.clf:
            self.logger.warn("Should have called train first!")
            return
        return self.clf.predict(datapoint)

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
            data (np.ndarray)
            labels (np.ndarray)
            k (int)
                How many ways to split the data
        Returns:
            (mean, std): tuple of floats
                the mean and standard deviation
                of cross validation.
        """
        if not self.clf:
            self.logger.error("Called cross validate"
                              "before training in"
                              "SvmLearner!")
            return
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import ShuffleSplit
        # make sure to shuffle the data!
        cv = ShuffleSplit(n_splits=k, test_size=.10)
        scores = cross_val_score(self.clf, data, labels, cv=cv)
        self.scores = scores
        return (scores.mean(), scores.std())
