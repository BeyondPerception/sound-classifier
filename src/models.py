from torch import nn, no_grad

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.network(x)

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2048, output_size),
        )
        self.init_weights()

    def init_weights(self):
        @no_grad()
        def init_weights_(m):
            init_range = 0.5
            if type(m) == nn.Linear:
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()
        self.network.apply(init_weights_)

    def forward(self, x):
        return self.network(x)
