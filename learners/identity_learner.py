from learners.abstract_learner import AbstractLearner


class IdentityLearner(AbstractLearner):
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
        feat = self.labeler.make_feature_vectors(labels, lab_to_index)
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

        # validation
        val_data_w_pkey, val_labels = self.labeler.make_matrix(validation_feat)
        self.val_data_w_pkey = val_data_w_pkey
        self.val_labels = val_labels
        val_data = val_data_w_pkey[1:, :]

        # get the signed distance for every train data point
        self.t_sd = t_data
        # for every validation data point
        self.val_sd = val_data

        return val_data, val_labels

    def predict(self, datapoint):
        """
        Args:
            datapoint (numpy.ndarray)
                Must have the same length as th
        Returns:
            signed distance from model
        """
        return datapoint
