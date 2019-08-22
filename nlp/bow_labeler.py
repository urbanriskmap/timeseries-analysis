import numpy as np
import os

from nlp.abstract_text_labeler import AbstractTextLabeler


class BowLabeler(AbstractTextLabeler):

    def __init__(self, config, loader):
        self.config = config
        self.logger = config["logger"]
        self.loader = loader
        super().__init__(config, loader)

    def load_labels_from_disk(self, filename='./bow_labels.p'):
        return super().load_labels_from_disk(filename)

    def dump_labels_to_disk(self, labels, filename='./bow_labels.p'):
        return super().dump_labels_to_disk(filename)

    def make_feature_vectors(self, reports_dict, vocab,
                             include_zero_vects=True):
        """
        Args:
            reports_dict: dict of {pkey: list(str)}
                From pkeys to a list of strings where
                each string is a token found in the text of pkey
            vocab: dict of {str: int}
                from token to index in the resulting vector
        Returns:
            feature_vect_dict: dict of {pkey, list(float)}
                A dictionary where each pkey corresponds to a
                feature vector for that pkey
        """
        feature_vect_dict = dict()

        flood = self.config["flood_pkeys"]
        no_flood = self.config["no_flood_pkeys"]
        remaining_pkeys = flood.union(no_flood)

        for pkey, word_list in reports_dict.items():
            if pkey in remaining_pkeys:
                remaining_pkeys.remove(pkey)
                feature_list = self.make_unary_feature_vector(vocab, word_list)
                feature_vect_dict[pkey] = feature_list

        if include_zero_vects:
            zero_list = [0]*len(vocab)
            for pkey in remaining_pkeys:
                feature_vect_dict[pkey] = zero_list

        return feature_vect_dict

    def get_labels(self, text_df, hook=None):
        """
        Args:
            text_df: (pd.DataFrame)
                Where the index is a pkey and
                there is at least one column called "text"
                that has a string.
        Returns:
            feat_vector_dict: dict of {pkey: list(str)}
                where each item is the tokens that are in
                the report text of each pkey
        """
        vocab, occur, feat_vector_dict = self.make_unary_vocab(text_df)
        self.vocab = vocab
        return feat_vector_dict

    def run_labeler(self, filename="bow_labels_default.p", rerun=False):
        """
        loads labels from disk or uses labeler api to labeles images if
        there are no labels on disk, then saves the labels to disk in
        config.data_folder_prefix/filename
        """
        # TODO: this should be derived from the base class, but
        #  there's slight differences so there's repeated code
        # between here and image_recognition/abstract_labeler
        label_path = os.path.join(self.config["data_folder_prefix"], filename)
        if rerun or not os.path.exists(label_path):
            texts_df = self.loader.get_texts()
            labels = self.get_labels(texts_df, self.dump_labels_to_disk)
        else:
            labels = self.load_labels_from_disk(filename)
        return labels

    def make_label_to_index(self, labels):
        """
        Args:
            Labels dict of (pkey: list(str))
        Returns:
            tuple (label_to_index, index_to_label)
            label_to_index: dict(str: int)
            index_to_label: dict(int: str)
        """
        # TODO: this is super inefficient
        if not self.vocab:
            text_df = self.loader.get_texts()
            vocab, occur, feat_vector_dict = self.make_unary_vocab(text_df)
            self.vocab = vocab
            self.logger.info("Should call"
                             "get_labels before"
                             "make_labels_to_index")
        # forwards is vocab,
        # lets make index_to_label
        index_to_label = dict([(index, current_label) for current_label, index
                               in self.vocab.items()])
        return (self.vocab, index_to_label)

    def make_unary_vocab(self, pkey_text_df, offset=0):
        """
            Params:
                pkey_text_df: a pandas DataFrame with an index
                              of pkey and a "text" column
                offset: where to begin indexing
            Returns:
                vocab(string: integer) - a dictionary from a
                                        word to an integer index
                occur(string: integer) - dictionary from word to the number
                                         of times it occurs in the corpus
                reports_dict(pkey: text)  - dictionary from pkey to report text
        """
        # go through all report texts creating set
        # of all possible words
        vocab = dict()
        occur = dict()
        index = offset
        reports_dict = dict()
        for row in pkey_text_df.iterrows():
            report_text_list = self.prepare_text(row[1]['text'])
            pkey = row[0]
            reports_dict[pkey] = report_text_list
            for word in report_text_list:
                if word not in vocab:
                    vocab[word] = index
                    index += 1
                if word not in occur:
                    occur[word] = 1
                else:
                    occur[word] += 1
        return (vocab, occur, reports_dict)

    def make_unary_feature_vector(self, vocab, report_text_list):
        res = [0]*len(vocab)
        for word in report_text_list:
            if word in vocab:
                res[vocab[word]] = 1
            # how to deal with out of vocab words?
        return res

    def make_feature_matrix(self,
                            reports_dict,
                            vocab,
                            make_vector=make_unary_feature_vector):
        """
        params:
            reports_dict: dictionary from pkeys to text
            vocab (dict): dictionary from word to index location along the
                      column vector
            make_vector: a function that takes a vocab and report_text_list and
                        returns a numpy feature vector
        Returns:
            tuple of (feature_matrix, labels)
            feature_matrix is a matrix of pkeys with associated feature columns
            underneath
            pkey      | pkey
            feat_vect | feat
            size: (len(vocab)+1, num_reports)
            labels is a matrix of size (1, num_reports) where report i is +1 if
                pkey matching that index is in pos_pkeys and -1 else
        """
        pos_pkeys = self.config['flood_pkeys']
        labels = np.zeros((1, (len(reports_dict))))
        feature_matrix = np.zeros((len(vocab)+1, len(reports_dict)))
        i = 0
        for pkey, word_list in reports_dict.items():
            col = make_vector(vocab, word_list)
            col_w_pkey = np.insert(col, 0, [pkey], 0)
            feature_matrix[:, i] = col_w_pkey[:, 0]
            if pkey in pos_pkeys:
                labels[0, i] = 1
            else:
                labels[0, i] = -1
            i += 1
        return (feature_matrix, labels)
