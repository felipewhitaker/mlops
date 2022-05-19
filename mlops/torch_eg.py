import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="{asctime} [{levelname}] {name} {message}",
    style="{",
    handlers=[logging.StreamHandler()],
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class Net(nn.Module):
    def __init__(self, input_: int, output_: int):
        super(Net, self).__init__()
        HIDDEN_SIZE = 4
        self.dense_input = torch.nn.Linear(input_, HIDDEN_SIZE)
        self.hidden = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.dense_output = torch.nn.Linear(HIDDEN_SIZE, output_)

        self.loss_fn = F.l1_loss

    def forward(self, X):

        X = torch.relu((self.dense_input(X)))
        X = torch.relu((self.hidden(X)))
        X = self.dense_output(X)

        return torch.sum(X, dim=1)  # F.sigmoid(X) # F.log_softmax(X,dim=1)

    @staticmethod
    def minmax_scaler(X):
        # FIXME sklearn Create a pipe with MinMaxScaler
        min_ = X.min()
        max_ = X.max()
        return (X - min_) / (max_ - min_)

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader):

        # TODO use pytorch lightning

        # FIXME allow for different optmizers
        # FIXME allow different learning rates
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)

        for e in range(epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                X, y = batch
                preds = self(self.minmax_scaler(X))
                loss = self.loss_fn(preds, y)
                loss.backward()
                optimizer.step()

            if e % 5 == 0:
                with torch.no_grad():
                    val_loss = self.eval(val_loader)
                logger.debug(f"{e=} loss={loss.item():.2f} {val_loss=:.2f}")

    def eval(self, loader=DataLoader):
        def validation_step(self, batch):
            X, y = batch
            with torch.no_grad():
                preds = self(X)
            loss = self.loss_fn(preds, y)
            return loss.detach()

        val_loss = (
            torch.stack(tuple(validation_step(self, batch) for batch in loader))
            .mean()
            .item()
        )
        return val_loss

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        return torch.load(path)

    def predict_one(self, values):
        with torch.no_grad():
            X = torch.Tensor(values)
            sX = self.minmax_scaler(X)
            nX = torch.atleast_2d(sX)
            return self(nX).item()


if __name__ == "__main__":

    # FIXME abstract load and train test split
    data = pd.read_csv("data/winequality-red.csv", nrows=30)
    X, y = data.drop(columns="quality").to_numpy(), data["quality"].to_numpy()
    X, y = torch.Tensor(X), torch.Tensor(y)

    dataset = TensorDataset(X, y)
    test_size = int(X.shape[0] * 0.3)
    train, test = random_split(dataset, [X.shape[0] - test_size, test_size])

    # FIXME allow changing batch size
    batch_size = 10
    train_loader = DataLoader(train, batch_size, shuffle=True)
    val_loader = DataLoader(test, batch_size)

    # create net
    clf = Net(X.shape[1], 1)
    # FIXME allow changing epochs
    clf.train(100, train_loader, val_loader)

    file_path = "models/dense"
    clf.save(file_path)
    logger.info(f"Saved model to {file_path}")
