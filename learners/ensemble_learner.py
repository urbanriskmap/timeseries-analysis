import torch
import numpy as np
import pickle
import os
import pandas as pd


import simple_nn as nn

# import flood_depth.flood_depth as flood_depth

from sklearn.base import BaseEstimator


class EnsembleLearner(BaseEstimator):

    def __init__(self, config, names, learners, hidden=10):
        """
        Args:
            learners: list of instances of PerceptronLearners
        """
        # self.config = config
        self.data_folder_prefix = config["data_folder_prefix"]
        # self.logger = config["logger"]
        self.learners = learners
        self.names = names
        self.hidden = hidden
        self.models = []

        self.flood = config["flood_pkeys"]
        self.no_flood = config["no_flood_pkeys"]
        self.all_pkeys = self.flood.union(self.no_flood)
        pass

    def _fill_no_data_spots(self, data_list, label_list):
        """
        Places zeros where data is missing
        Args:
            data_list: list of ndarrays where the first row is
                pkeys and the second row is a float datapoint
            labels: ndarray of labels
        Returns:
            pd.DataFrame:
        """
        all_pkeys = set()
        for each in data_list:
            all_pkeys.update(each[0, :].tolist())
        res = pd.DataFrame(all_pkeys,
                           columns=["pkey"]).set_index("pkey",
                                                       drop=True).sort_index()
        last_index = 0
        lab_df = pd.DataFrame(all_pkeys,
                              columns=["pkey"]).set_index("pkey",
                                                          drop=True).sort_index()
        lab_df["label"] = 0
        for i, (data, labels) in enumerate(zip(data_list, label_list)):
            vect_len = data.shape[0]-1
            # pkeys should be index in the first row of data
            ind = pd.Index(data[0, :], name="pkey")
            add_df = pd.DataFrame(data=data[1:, :].T, index=ind, columns=range( last_index, last_index+vect_len))
            last_index += vect_len
            res = pd.concat([res, add_df], axis=1).fillna(0)
            labs = pd.DataFrame(data=labels.T, columns=["label"], index=ind)
            lab_df.update(labs)
        return res.join(lab_df).fillna(0)

    def fit(self, X, y):
        """
        Args:
            X (np.array) of [n_samples, n_features]
            y (np.array) of [n_samples]
                where each member is -1 or +1
        Returns:
            self (object)
        """
        t_full_matrix = torch.from_numpy(X).float()
        # sklearn wants -1, 1 class labels but torch expects 0, 1
        into_zeros = np.where(y < 0, 0, 1)
        t_full_labels = torch.from_numpy(into_zeros).long()
        self.nn_model = nn.Simple_nn(X.shape[1], self.hidden)
        nn.run_training(self.nn_model, t_full_matrix, t_full_labels)
        return self

    def train(self, params, validation_keys):
        """
        Runs all learners

        """
        all_pkeys = self.flood.union(self.no_flood)
        assert(validation_keys.issubset(all_pkeys))

        # run all the learners
        train_sd = []
        val_sd = []
        t_labels_list = []
        val_labels_list = []
        self.val_labels = []
        for name, each in zip(self.names, self.learners):
            model = each.run_learner(name,
                                     rerun=False,
                                     validation_keys=validation_keys,
                                     params=params)
            self.models.append(model)

            t_sd_w_pkey = np.vstack((each.t_data_w_pkey[0, :],
                                    each.t_sd))
            train_sd.append(t_sd_w_pkey)
            t_labels_list.append(each.t_labels)

            val_sd_w_pkey = np.vstack((each.val_data_w_pkey[0, :],
                                      each.val_sd))
            val_sd.append(val_sd_w_pkey)
            val_labels_list.append(each.val_labels)

        # arrange a matrix st each column is the
        # result of predicting on this pkey
        # ex: the signed distance from the separator
        self.train_pd = self._fill_no_data_spots(train_sd,
                                                 t_labels_list)
        t_with_labels = self.train_pd.to_numpy().T
        self.t_labels = t_with_labels[-1, :]
        self.train_matrix = t_with_labels[:-1, :]  # remove label
        # self.logger.info("training size " + str(self.train_matrix.shape))
        # self.logger.info("training matrix " + str(self.train_matrix))

        self.val_pd = self._fill_no_data_spots(val_sd, val_labels_list)
        val_w_labels = self.val_pd.loc[pd.Index(validation_keys)].to_numpy().T
        # take out the labels
        self.val_labels = val_w_labels[-1, :]
        self.val_matrix = val_w_labels[:-1, :]
        # self.logger.info("validation size " + str(self.val_matrix.shape))

        self.hidden_layers = params["hidden"]

        self.fit(self.train_matrix.T, self.t_labels)
#         t_full_matrix = torch.from_numpy(train_matrix.T).float()
#         # no flood is zero class, flood is 1st class
#         into_zeros = np.where(self.t_labels < 0, 0, 1)
#         t_full_labels = torch.from_numpy(into_zeros).long()
#         nn_model = nn.Simple_nn(len(self.models), self.hidden_layers)
#         self.nn_model = nn_model
#         # nn_model = nn.Simple_nn(len(self.models), hidden_layers)
#         nn.run_training(nn_model, t_full_matrix, t_full_labels)

        torch_val_matrix = torch.from_numpy(self.val_matrix.T).float()
        self.res = self.nn_model(torch_val_matrix)
        return self.nn_model

    def run_learner(self,
                    filename,
                    rerun=False,
                    validation_keys=dict(),
                    params={"T": 1000, "print": True}):
        path = os.path.join(self.data_folder_prefix, filename)
        if rerun or not os.path.exists(path):
            nn_model = self.train(params, validation_keys=validation_keys)
            self.dump_model_to_disk(nn_model, filename)
            return nn_model
        else:
            nn_model = self.load_model_from_disk(filename)
        return nn_model

    def dump_model_to_disk(self, model, filename="ensemble_learner_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        # self.logger.debug("dumping model to: " + str(path))
        pickle.dump(model, open(path, "wb"))
        return

    def predict(self, X):
        """
        Args:
            X: (ndarray) of [n_samples, n_features]

        Returns:
            y:
                +1, -1 label
        """
        datapoint = torch.from_numpy(X).float()
        import math
        logSoftmaxOutput = self.nn_model(datapoint)
        probs = math.e**logSoftmaxOutput
        p = probs.data.numpy()

        predicted = np.argmax(p, axis=1)  # which index is greater
        # now from index to -1, 1
        to_class_label = np.where(predicted == 0, -1, 1)
        return to_class_label

    def load_model_from_disk(self, filename="perceptron_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        # self.logger.debug("loading from: " + str(path))
        return pickle.load(open(path, "rb"))
