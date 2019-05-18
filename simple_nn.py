import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

import image_recognition.aws_recog as aws

class Simple_nn(torch.nn.Module):
    def __init__(self, dims_in, hidden):
        super(Simple_nn, self).__init__()
        self.linear1 = torch.nn.Linear(dims_in, hidden)
        self.linear2 = torch.nn.Linear(hidden, 2)
        self.output = torch.nn.LogSoftmax()

    def forward(self, x):
        hidden_activation = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(hidden_activation)
        return self.output(y_pred)


def run_training(model, x_data, y_data):
    lossfn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    for epoch in range(2000):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_data)
        # Compute Loss
        loss = lossfn(y_pred, y_data)
        print('Loss:', loss)
    
        # Backward pass
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    labels = aws.read_labels_from_disk()
    clean = aws.clean_if_dirty(labels)

    all_labels = set()
    for key, each in clean.items():
        for lab in each['Labels']:
            if lab['Name'] not in all_labels:
                all_labels.add(lab['Name'])
        
    print(len(all_labels))


    # allowed has label: index pairs for ex: 'Flood':0, 
    # so 'Flood' is the first 
    # in the feature vector 
    #allowed = dict([ (current_label, index) for index, current_label in enumerate(list(all_labels))])
    more = ['Flood']
    allowed = dict([ (current_label, index) for index, current_label in enumerate(list(more))])

    vects = aws.make_feature_vectors(clean, allowed)

    matrix_w_pkey = aws.make_matrix_rep(vects, len(list(allowed)))
    labels_w_pkey = aws.make_labels_rep(vects)

    matrix = torch.from_numpy(matrix_w_pkey[1:,:].T).float()
    labels = torch.from_numpy(labels_w_pkey[1:,:].T).float()
    # matrix = torch.from_numpy(np.vstack((matrix_w_pkey[1:,:], labels_w_pkey[1:,:])).T).float()

    run_training(matrix, labels)
