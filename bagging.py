import torch
import numpy as np
import pickle

import image_recognition.aws_recog as aws_recog
import image_recognition.perceptron as perceptron
import img_util as img_util

import simple_nn as nn

import nlp.aws_nlp as aws_nlp
import nlp_util as nlp_util

import flood_depth.flood_depth as flood_depth

SEPARATOR_FILENAME = './separator.p'
USE_SAVED_SEPARATOR = False

LOCATION = 'ch'
# LOCATION = 'id'

def load_saved_separator():
    return pickle.load(open(SEPARATOR_FILENAME,'rb'))

def cross_validation(data, label, k=5):
    # split the data!
    '''
    returns: train_data, train_label, test_data, test_label
    '''
    full = np.vstack((data,label)).copy()
    np.random.shuffle(full.T)
    data = full[:-1, :]
    labels = full[-1:, :]

    # axis=1 means columns, split into groups of cols
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)

    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])

        t_data_train = torch.from_numpy(data_train.T).float()
        t_labels_train = torch.from_numpy(np.where(labels_train<0, 0, 1)[0,:]).long()

        new_model = nn.Simple_nn(data_test.shape[0], 1)
        nn.run_training(new_model, t_data_train, t_labels_train)

        t_data_test = torch.from_numpy(data_test.T).float()
        t_labels_test = torch.from_numpy(np.where(labels_test<0, 0, 1)).long()

        predicted = torch.exp(new_model(t_data_test)).data.numpy()
        # if the first column is less, then 0 (no flood)
        # else there is flooding, 1
        # multiplying by one turns it from bool to int
        pred = 1*(predicted[:,0] < predicted[:, 1] )
        # print(np.where(labels_test<0, 0, 1))
        # print(pred == np.where(labels_test<0, 0, 1))
        score_sum += np.sum(pred == np.where(labels_test<0, 0, 1))/labels_test.shape[1]
        print('score_sum', score_sum)
    return score_sum/k, score_sum

if __name__ == '__main__':
    if (USE_SAVED_SEPARATOR):
        th, th0, data_w_pkey, labels_w_pkey = load_saved_separator()
    else:
        if LOCATION == 'ch':
            # th, th0, data_w_pkey, labels_w_pkey = img_util.perf_given_categories( img_util.TOP10_CH, label_filename='./image_recognition/min_confidence_chennai_labels.p', location='ch')
            th, th0, data_w_pkey, labels_w_pkey = img_util.perf_given_categories( img_util.top30allowed, label_filename='./image_recognition/min_confidence_chennai_labels.p', location='ch')
        else:
            th, th0, data_w_pkey, labels_w_pkey = img_util.perf_given_categories(img_util.ALL_LABELS)

        # save it! 
        pickle.dump((th, th0, data_w_pkey, labels_w_pkey), open(SEPARATOR_FILENAME, 'wb'))

    # remove the pkeys, still ordered by index can look up later
    labels = labels_w_pkey[1:, :]
    data = data_w_pkey[1:,:] 

    t_labels = torch.from_numpy(np.where(labels<0, 0, 1)[0,:]).long()
    t_data = torch.from_numpy(data.T).float()

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

    # print(sents_matrix)
    # load flood heights
    if (LOCATION == 'ch'):
        flood_depth_df = flood_depth.get_flood_depth_chennai()
    else:
        flood_depth_df = flood_depth.get_flood_depth_jakarta()

    flood_depth_w_pkeys = flood_depth.make_matrix(flood_depth_df)
    
    full_matrix_w_pkeys = np.zeros((4, data_w_pkey.shape[1]))
    full_matrix_w_pkeys[0,:] = data_w_pkey[0,:]
    
    full_labels_w_pkeys = np.zeros((2, labels_w_pkey.shape[1]))
    full_labels_w_pkeys[0,:] = labels_w_pkey[0,:]
    n = full_matrix_w_pkeys.shape[1]

    for i in range(n):
        # get the signed dist
        full_matrix_w_pkeys[1, i] = sd[i, 0]
        # get neg sent
        full_matrix_w_pkeys[2, i] = sents_matrix_w_pkey[1, i]
        full_matrix_w_pkeys[3, i] = flood_depth_w_pkeys[1, i]

    flood_depth_matrix = flood_depth_w_pkeys[1:,:]
    full_matrix = full_matrix_w_pkeys[1:, :]
    full_labels = full_labels_w_pkeys[1:, :]

    t_full_matrix = torch.from_numpy(full_matrix.T).float()
    t_full_labels = torch.from_numpy(np.where(full_labels<0, 0, 1)[0,:]).long()

    trials = 100
    k = 5
    s = 0
    for n in range(trials):
        score, s_sum = cross_validation(full_matrix, full_labels, k=k)
        s += s_sum
        print(score)

    print("overall score:")
    print(s/(k*trials))

#    model = nn.Simple_nn(3, 1)
#    nn.run_training(model, t_full_matrix, t_full_labels)
