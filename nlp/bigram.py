import numpy as np


class bigram_labeler():

    def __init__(self, config):
        self.config = config
        super().__init__(config)

    def load_labels_from_disk(self, filename='./bigram_labels.p'):
        return super().load_labels_from_disk(filename)

    def dump_labels_to_disk(self, labels, filename='./bigram_labels.p'):
        return super().dump_labels_to_disk(filename)

    def get_labels(self):
        self.config.loader
        text_data = self.loader.get_data()
        return self.make_bigram_vocab(text_data)

    def make_bigram_vocab(self, pkey_text_df, offset=0):
        """
        """
        vocab = dict()
        occur = dict()
        reports_dict = dict()
        index = offset
        for row in pkey_text_df.iterrows():
            report_text_list = self.prepare_text(row[1]['text'])
            pkey = row[0]
            two_gram = list(zip(report_text_list, report_text_list[1:]))
            reports_dict[pkey] = two_gram
            for first, second in two_gram:
                token = (first, second)
                if token not in vocab:
                    vocab[token] = index
                    index += 1
                if token in occur:
                    occur[token] += 1
                else:
                    occur[token] = 1
        return (vocab, occur, reports_dict)

    def make_bigram_feature_vector(self, vocab, report_text_list):
        # needs a tuple vocab[(tup)] -> index
        res = np.zeros((len(vocab), 1))
        for first, second in report_text_list:
            if (first, second) in vocab:
                res[vocab[(first, second)]][0] = 1
            # how to deal with out of vocab words?
        return res
