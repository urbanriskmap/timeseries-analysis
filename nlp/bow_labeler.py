import numpy as np

from abstract_labeler import AbstractTextLabeler


class BowLabeler(AbstractTextLabeler):

    def __init__(self, config, loader):
        self.config = config
        self.loader = loader
        super().__init__(config)

    def load_labels_from_disk(self, filename='./bow_labels.p'):
        return super().load_labels_from_disk(filename)

    def dump_labels_to_disk(self, labels, filename='./bow_labels.p'):
        return super().dump_labels_to_disk(filename)

    def make_feature_vectors(self, inp, allowed):
        """
        Args:
            inp: dict of {pkey: list(str)}
                From pkeys to a list of strings where
                each string is a token found in the text of pkey
            allowed: dict of {str: int}
                from token to index in the resulting vector
        Returns:
            feature_vect_dict: dict of {pkey, list(float)}
                A dictionary where each pkey corresponds to a
                feature vector for that pkey
        """

        pass

    def get_labels(self, text_df, hook=None):
        """
        Args:
            text_df: (pd.DataFrame)
                Where the index is a pkey and
                there is at least one column called "text"
                that has a string.
        Returns:
            labels: dict of {pkey: list(str)}
        """
        self.loader
        text_data = self.loader.get_texts()
        vocab, occur, reports = self.make_unary_vocab(text_data)
        return reports

    def make_labels_to_index(self, labels):
        pass

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
        res = np.zeros((len(vocab), 1))
        for word in report_text_list:
            if word in vocab:
                res[vocab[word]][0] = 1
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
