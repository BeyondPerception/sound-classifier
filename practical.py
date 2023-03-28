import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Load train data

data_dir = "data/"

num_classes = 10

X_amp_train = np.load(data_dir + "Xtrain_amp.npy")
y_amp_train = np.load(data_dir + "ytrain_amp.npy")

# Load test data

X_amp_test = np.load(data_dir + "Xtest_amp.npy")
y_amp_test = np.load(data_dir + "ytest_amp.npy")

print("Train data X shape:", X_amp_train.shape)
print("Test data X shape:", X_amp_test.shape)

input_size = len(X_amp_train[0])
output_size = num_classes

# # Load melSpectrogram train data
# X_mel_train = np.load(data_dir + "Xtrain_mel.npy")
# y_mel_train = np.load(data_dir + "ytrain_mel.npy")

# # Flatten X_mel_train's spectrogram features
# X_mel_train_flat = X_mel_train.reshape(X_mel_train.shape[0], -1)
# X_mel_train_flat.shape

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_size),
        )
        self.init_weights()

    def init_weights(self):
        @torch.no_grad()
        def init_weights_(m):
            init_range = 0.5
            if type(m) == nn.Linear:
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()
        self.network.apply(init_weights_)

    def forward(self, x):
        return self.network(x)

# Hyperparameters
EPOCHS = 3 # epoch
LR = 5  # learning rate
BATCH_SIZE = 1 # batch size for training

zipped_data = list(zip(X_amp_train, y_amp_train))
dataloader = DataLoader(zipped_data, batch_size=BATCH_SIZE, shuffle=False)

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

def train(dataloader):
    model.train()

    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (amps, label) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(amps[0])
        loss = criterion(predicted_label, label[0])
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax() == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (amps, label) in enumerate(dataloader):
            predicted_label = model(amps[0])
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

print("Starting to train...")

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(dataloader)
    scheduler.step()
    print(f"Epoch {epoch} took {time.time() - epoch_start_time:0.2f} seconds")

torch.save(model, "basic.model")

zipped_data_test = list(zip(X_amp_test, y_amp_test))
test_dataloader = DataLoader(zipped_data_test, batch_size=BATCH_SIZE, shuffle=False)

acc = evaluate(test_dataloader)
print(f"Model accuracy: {acc}")
