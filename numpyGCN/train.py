from __future__ import print_function

import json
import time
import argparse
from datetime import datetime
import numpy as np

# Set random seed
seed = 42
#np.random.seed(seed)

from numpyGCN import numpyGCN
from utils import load_data

def train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask, early_stopping, lr, epochs):
    t_total = time.time()
    best_val_loss, val_epoch = float('inf'), 0
    past_loss = float('inf')
    for epoch in range(epochs):
        start = time.time()
        model.gd_update(features, y_train, adj, train_mask, lr)
        end = time.time()

        train_loss, train_accuracy = model.loss_accuracy(features, y_train, adj, train_mask)
        val_loss, val_accuracy = model.loss_accuracy(features, y_val, adj, val_mask)

        elapsed = end - start
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_accuracy), "time=", "{:.5f}".format(elapsed))

        # decrease the learning rate if the train loss increased
        if train_loss > past_loss:
            lr *= 0.5
        train_loss = min(train_loss, past_loss)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_epoch = epoch
            else:
                if epoch - val_epoch == 10:
                    print("validation loss has not improved for 10 epochs... stopping early")
                    break

    print("Total time: {:.4f}s".format(time.time() - t_total))

def train(learning_rate, early_stopping, dropout, weight_decay, epochs, dataset):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)

    model = numpyGCN(
        input_dim=features.shape[1],
        hidden_dim=16,
        output_dim=y_train.shape[1],
        dropout=dropout,
        weight_decay = weight_decay
    )
    train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask,
        early_stopping=early_stopping, lr=learning_rate, epochs=epochs)

    test_loss = model.calc_loss(features, y_test, adj, test_mask)
    test_accuracy = model.compute_accuracy(features, y_test, adj, test_mask)

    print("test_loss=", "{:.5f}".format(test_loss), "test_acc=", "{:.5f}".format(test_accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("numpyGCN")
    parser.add_argument('--dataset', default='cora')
    args = parser.parse_args()

    print('Using {} dataset'.format(args.dataset))

    with open('numpyGCN_hyperparams.json') as f:
        hp = json.load(f)

    lr = hp['learning_rate']
    d = hp['dropout']
    w_d = hp['weight_decay']
    train(learning_rate=lr, early_stopping=True, dropout=d, weight_decay=w_d, epochs=200, dataset=args.dataset)
