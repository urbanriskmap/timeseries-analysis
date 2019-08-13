import numpy as np


def prepare_text(self, inp):
    '''
    returns a list of strings where each string is a different token
    '''
    # TODO replace this with regex?
    import string
    exclude = set(string.punctuation)
    s = "".join(ch for ch in inp if ch not in exclude)
    return s.lower().replace('\n', ' ').split()


def make_feature_matrix(self,
                        reports_dict,
                        vocab,
                        make_vector):
    """
    params:
        report_texts: dictionary from pkeys to text
        vocab (dict): dictionary from word to index location along the
                  column vector
        pos_pkeys (set): membership in this set means the pkey should be
                         labeled as postive
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
    print(self.config)
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
