import torch
import numpy as np
import pickle
import os

# import img_util as img_util

import simple_nn as nn

# import flood_depth.flood_depth as flood_depth


class EnsembleLearner():

    def __init__(self, config, img_learn, text_learn):
        """
        Args:
            learners: list of instances of PerceptronLearners
        """
        self.config = config
        self.data_folder_prefix = config["data_folder_prefix"]
        self.logger = config["logger"]
        self.img_learn = img_learn
        self.text_learn = text_learn
        self.learners = [img_learn, text_learn]
        self.models = []
        self.names = ["img_model.p", "text_model.p"]
        # self.learners = [img_learn, text_learn, height_learn]
        # self.names = ["img_model.p", "text_model.p", "height_model.p"]
        pass

    def train(self, params, validation_keys):
        """
        Runs all learners

        """
        flood = self.config["flood_pkeys"]
        no_flood = self.config["no_flood_pkeys"]
        all_pkeys = flood.union(no_flood)

        assert(validation_keys.issubset(all_pkeys))

        # run all the learners
        train_sd = []
        val_sd = []
        t_labels = []
        self.val_labels = []
        for name, each in zip(self.names, self.learners):
            model = each.run_learner(name,
                                     rerun=True,
                                     validation_keys=validation_keys,
                                     params=params)
            self.models.append(model)
            train_sd.append(each.t_sd)
            val_sd.append(each.val_sd)
            t_labels = each.t_labels
            self.val_labels = each.val_labels

        # arrange a matrix st each column is the
        # result of predicting on this pkey
        # ex: the signed distance from the separator
        train_matrix = np.vstack(train_sd)
        val_matrix = np.vstack(val_sd)

        t_full_matrix = torch.from_numpy(train_matrix.T).float()
        # no flood is zero class, flood is 1st class
        into_zeros = np.where(t_labels < 0, 0, 1)[0, :]
        t_full_labels = torch.from_numpy(into_zeros).long()

        hidden_layers = params["hidden"]
        nn_model = nn.Simple_nn(len(self.models), hidden_layers)
        nn.run_training(nn_model, t_full_matrix, t_full_labels)

        torch_val_matrix = torch.from_numpy(val_matrix.T).float()
        self.res = nn_model(torch_val_matrix)
        return nn_model

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
        self.logger.debug("dumping model to: " + str(path))
        pickle.dump(model, open(path, "wb"))
        return

    def load_model_from_disk(self, filename="perceptron_default.p"):
        path = os.path.join(self.data_folder_prefix, filename)
        self.logger.debug("loading from: " + str(path))
        return pickle.load(open(path, "rb"))
