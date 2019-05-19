import torch
import numpy as np
import pickle

import image_recognition.aws_recog as aws_recog
import image_recognition.perceptron as perceptron
import img_util as img_util

import simple_nn as nn

import nlp.aws_nlp as aws_nlp
import nlp_util as nlp_util 

import flood_depth as flood_depth

SEPARATOR_FILENAME = './separator.p'
USE_SAVED_SEPARATOR = True

LOCATION = 'ch'

def load_saved_separator():
    return pickle.load(open(SEPARATOR_FILENAME,'rb'))


if __name__ == '__main__':
    if (USE_SAVED_SEPARATOR):
        th, th0, data_w_pkey, labels_w_pkey = load_saved_separator()
    else: 
        th, th0, data_w_pkey, labels_w_pkey = img_util.perf_given_categories(img_util.ALL_LABELS)
        # save it! 
        pickle.dump((th, th0, data_w_pkey, labels_w_pkey), open(SEPARATOR_FILENAME, 'wb'))

    # remove the pkeys, still ordered by index can look up later
    labels = labels_w_pkey[1:, :]
    data = data_w_pkey[1:,:] 

    t_labels = torch.from_numpy(np.where(labels<0, 0, 1)[0,:]).long()
    t_data = torch.from_numpy(data.T).float()
    print(t_data.shape)

    # the signed distance from the separator for every point
    sd = (np.dot(data.T, th) + th0)
    # each row is a datapoint
    t_sd = torch.from_numpy(sd).float()

    # add in the sentimets 
    # only for chennai! 
    if (LOCATION == 'ch'):
        sents = nlp_util.load_pickled_sents()
        sents_matrix_w_pkey = nlp_util.build_matrix(sents)
        sents_matrix = sents_matrix_w_pkey[1:,:]
        print(sents_matrix_w_pkey.shape)

    # load flood heights
    if (LOCATION == 'ch'):
        flood_depth_df = flood_depth.get_flood_depth_chennai()
    else:
        flood_depth_df = flood_depth.get_flood_depth_jakarta()

    flood_depth_w_pkeys = flood_depth.make_matrix(flood_depth_df)
    flood_depth_matrix = flood_depth_w_pkeys[1:,:]

    full_matrix = np.vstack((sd.T, flood_depth_matrix, sents_matrix))
    model = nn.Simple_nn(3, 1)

    nn.run_training(model, t_sd, t_labels)

