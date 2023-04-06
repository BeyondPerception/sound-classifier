#!/usr/bin/env python

import time

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models import LogisticRegression, SimpleNeuralNetwork

# Use gpu if possible
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

num_classes = 10

# Load train data
data_dir = "../data/"

X_amp_train = np.load(data_dir + "Xtrain_amp.npy")
y_amp_train = np.load(data_dir + "ytrain_amp.npy")

# Load melSpectrogram train data
X_mel_train = np.load(data_dir + "Xtrain_mel.npy")
y_mel_train = np.load(data_dir + "ytrain_mel.npy")

# Load test data
X_amp_test = np.load(data_dir + "Xtest_amp.npy")
y_amp_test = np.load(data_dir + "ytest_amp.npy")

# Load melSpectrogram test data
X_mel_test = np.load(data_dir + "Xtest_mel.npy")
y_mel_test = np.load(data_dir + "ytest_mel.npy")

# Flatten X_mel_train's spectrogram features
X_mel_train_flat = X_mel_train.reshape(X_mel_train.shape[0], -1)
X_mel_test_flat = X_mel_test.reshape(X_mel_test.shape[0], -1)

def train(model, dataloader, criterion, optimizer):
    model.train()

    total_acc, total_count = 0, 0
    log_interval = 20
    start_time = time.time()

    for idx, (amps, labels) in enumerate(dataloader):
        # Move tensors to gpu.
        Xs = amps.to(device)
        labels = labels.to(device)

        # Run SGD.
        optimizer.zero_grad()
        predicted_labels = model(Xs)
        loss = criterion(predicted_labels, labels)
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        # Print accuracy for the batch.
        total_acc += (predicted_labels.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            # print(list(model.parameters()))
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (amps, labels) in enumerate(dataloader):
            Xs = amps.to(device)
            labels = labels.to(device)
            predicted_labels = model(Xs)
            loss = criterion(predicted_labels, labels)
            total_acc += (predicted_labels.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc/total_count

if __name__ == '__main__':
    needs_training = False

    # Test logistic regression on standard amp data.
    if False:
        X_input_train = X_amp_train
        y_input_train = y_amp_train
        X_input_test = X_amp_test
        y_input_test = y_amp_test

        print("Train data X shape:", X_input_train.shape)
        print("Test data X shape:", X_input_test.shape)

        input_size = X_input_train.shape[1]
        output_size = num_classes

        model = LogisticRegression(input_size, output_size).to(device)
        needs_training = False

    # Test logistic regression on mel spectrogram data.
    if False:
        X_input_train = X_mel_train_flat
        y_input_train = y_mel_train
        X_input_test = X_mel_test_flat
        y_input_test = y_mel_test

        print("Train data X shape:", X_input_train.shape)
        print("Test data X shape:", X_input_test.shape)

        input_size = X_input_train.shape[1]
        output_size = num_classes

        model = LogisticRegression(input_size, output_size).to(device)
        needs_training = True

    # Test simple neural network on standard amp data.
    if True:
        X_input_train = X_amp_train
        y_input_train = y_amp_train
        X_input_test = X_amp_test
        y_input_test = y_amp_test

        print("Train data X shape:", X_input_train.shape)
        print("Test data X shape:", X_input_test.shape)

        input_size = X_input_train.shape[1]
        output_size = num_classes

        model = SimpleNeuralNetwork(input_size, output_size).to(device)
        needs_training = True

    # Test simple neural network on mel spectrogram data.
    if False:
        X_input_train = X_mel_train_flat
        y_input_train = y_mel_train
        X_input_test = X_mel_test_flat
        y_input_test = y_mel_test

        print("Train data X shape:", X_input_train.shape)
        print("Test data X shape:", X_input_test.shape)

        input_size = X_input_train.shape[1]
        output_size = num_classes

        model = SimpleNeuralNetwork(input_size, output_size).to(device)
        needs_training = True

    BATCH_SIZE = 64 # batch size for training
    criterion = nn.CrossEntropyLoss()

    # Train the model
    if (needs_training):
        print("Starting to train...")

        # Hyperparameters
        EPOCHS = 10 # num epochs
        LR = 0.001  # learning rate

        zipped_data = list(zip(X_input_train, y_input_train))
        train_dataloader = DataLoader(zipped_data, batch_size=BATCH_SIZE, shuffle=True)

        optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.01)

        for epoch in range(1, EPOCHS + 1):
            train(model, train_dataloader, criterion, optimizer)
            # scheduler.step()

        torch.save(model, "basic.model")
        
    # model = torch.load("../linear_reg_amp.model").to(device)

    # Evaluate the model
    zipped_data_test = list(zip(X_input_test, y_input_test))
    test_dataloader = DataLoader(zipped_data_test, batch_size=BATCH_SIZE, shuffle=False)

    acc = evaluate(model, test_dataloader, criterion)
    print(f"Model accuracy: {acc:0.3f}")


# print(list(model.parameters()))

# (amps, label) = next(iter(dataloader))

# amps = amps.to(device)
# label = label.to(device)
# predicted_label = model(amps)

# print(amps)
# print(label)
# print(predicted_label)
