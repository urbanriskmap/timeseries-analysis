import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

import aws_recog as aws

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearRegression()


def run_training(x_data, y_data):
    lossfn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000000001)
    for epoch in range(20000):
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

    predicted =model.forward(Variable(x_data[0]))
    print('params: ')
    for param in model.parameters():
          print(param.data)

    print(predicted)

    print(predicted.size())


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
