import torch


class Simple_nn(torch.nn.Module):
    def __init__(self, dims_in, hidden):
        super(Simple_nn, self).__init__()
        self.linear1 = torch.nn.Linear(dims_in, hidden)
        self.linear2 = torch.nn.Linear(hidden, 2)
        self.output = torch.nn.LogSoftmax()

    def forward(self, x):
        hidden_activation = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(hidden_activation).clamp(min=0)
        return self.output(y_pred)


def run_training(model, x_data, y_data, num_epochs=10000):
    lossfn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0015)
    n = 20
    last_n = [0]*n
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_data)
        # Compute Loss
        loss = lossfn(y_pred, y_data)
        print('Loss:', loss)
        if epoch > n:
            # early stopping
            last_n.pop(0)
            last_n.append(loss)
            diff = abs(last_n[-1] - last_n[0])
            if diff < .0001:
                print("Early stopping at epoch: " + str(epoch))
                break

        # Backward pass
        loss.backward()
        # limit the size to prevent
        # exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), 100)
        optimizer.step()
