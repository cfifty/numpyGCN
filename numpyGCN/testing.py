from __future__ import print_function

import time
from datetime import datetime
import numpy as np

from numpyGCN import numpyGCN
from utils import load_data

# test the forward pass
def test_forward():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
    model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])

    preds = model.predict(features, adj)

    print("preds.shape " + str(preds.shape))
    print("accuracy: " + str(model.compute_accuracy(features, y_test, adj, test_mask)))

def test_calc_loss():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
    model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])

    loss = model.calc_loss(features, y_train, adj, train_mask)
    print("cross entropy loss " + str(loss))

# test one gradient descent update
def test_gd_step():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
    model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])
    model.gd_update(features, y_train, adj, train_mask, lr=0.1, d=0.2, w_d=0.0)

if __name__ == '__main__':
    test_gd_step()
    test_calc_loss()
    test_forward()