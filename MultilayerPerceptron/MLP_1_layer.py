import os

import torch
import torch.nn as nn
import random

import time

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return (x, y)

    def __len__(self):
        count = self.x.shape[0]
        return count


class Perceptron_pytorch(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob):
        super(Perceptron_pytorch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_1 = nn.Linear(self.input_size, self.hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu_1 = torch.nn.ReLU()
        self.dropout_1 = nn.Dropout(p=dropout_prob)
        self.fc_2 = nn.Linear(self.hidden_size, 1)
    def forward(self, x):
        x = self.fc_2(self.dropout_1(self.relu_1(self.bn1(self.fc_1(x)))))
        return x
    def init_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Perceptron():
    def __init__(self, epochs=500):
        self.model = None
        self.epochs = epochs
        self.epoch = 0
    def fit(self, Xtrain, ytrain, Xval, yval, fold, input_size, hidden_size, lr=0.01, weight_decay=1e-4, dropout=0.1, pretrained_model_path=None):
        X_val = (torch.tensor(Xval)).float()
        y_val = torch.tensor(yval).float()
        X = (torch.tensor(Xtrain)).float()
        y = torch.tensor(ytrain).float()

        self.y_span = max(y)-min(y)
        self.y_lower = min(y)

        if self.model is None:
            model = Perceptron_pytorch(input_size, hidden_size, dropout)
            if pretrained_model_path:
                model.load_state_dict(torch.load(pretrained_model_path))
                print(f"Loaded pretrained model from {pretrained_model_path}")
            else:
                model.init_params()
            self.model = model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # crit = nn.L1Loss()
        crit = nn.MSELoss()
        # crit = nn.HuberLoss(reduction='mean', delta=5.0)

        best_epoch = 0
        best_mae_val = np.inf
        val_improve_epoch = 0
        epochs_wo_improve = 50
        total_updates = 0
        avg_loss = []

        train_loss_list, val_loss_list, epoch_list = [], [], []

        for epoch in range(self.epochs):

            train_dataset = Dataset(X, y)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=int(X.shape[0] / int(X.shape[0]/64)), shuffle=True, drop_last=True)
            total_train_loss = 0.0

            for train_data in train_loader:

                # calculo la pred y loss con datos de batch
                model.train()
                y_pred = torch.sigmoid((model(train_data[0])).squeeze()) * self.y_span + self.y_lower
                loss = crit(y_pred.reshape(-1, 1), train_data[1].reshape(-1, 1))

                total_train_loss += loss.item()

                # guardo la loss por batch
                avg_loss.append(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_updates += 1

            # obtengo resultado validación
            model.eval()
            with torch.no_grad():
                y_pred_val = torch.sigmoid((model(X_val)).squeeze()) * (max(y)-min(y)) + min(y)

            mae_val = mean_squared_error(y_val.detach().numpy(), y_pred_val.detach().numpy())

            if (mae_val <= best_mae_val) or best_epoch == 0:
                if mae_val < best_mae_val:
                    val_improve_epoch = epoch
                best_mae_val = mae_val
                best_epoch = epoch
                best_model = model
                print('model actualized')

            # no improvement
            if epoch - val_improve_epoch >= epochs_wo_improve:
                break

            train_loss_list.append(total_train_loss / len(train_loader))
            val_loss_list.append(mae_val)
            epoch_list.append(epoch)

            print('model, epoch: {}, mae_train: {}, mae_val: {}'.format(epoch, total_train_loss / len(train_loader), mae_val))

        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, train_loss_list, label='Training Loss', marker=None, color='blue')
        plt.plot(epoch_list, val_loss_list, label='Validation Loss', marker=None, color='red')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Saving the plot as a PNG file
        plt.savefig('training_validation_loss_' + str(fold) + '.svg')

        print('no improvement in validation in the last 20 epochs, returning best model, epoch: {}, best_mae_val: {}'.format(best_epoch, best_mae_val))
        return best_model


    def predict(self, X):
        X = (torch.tensor(X)).float()
        result = torch.sigmoid((self.model(X)).squeeze()) * self.y_span + self.y_lower
        prediction = result.cpu().detach().numpy()
        return prediction.squeeze()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





