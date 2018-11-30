import time
from datetime import datetime
import numpy as np

from numpyGCN import numpyGCN
from utils import load_data

def train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask, early_stopping=True, lr=0.005, epochs=200):
    total_time = 0
    best_val_loss, val_epoch = float('inf'), 0
    for epoch in range(epochs):
        start = time.time()
        model.gd_update(features, y_train, adj, train_mask, lr=0.1)
        end = time.time()
        train_loss = model.calc_loss(features, y_train, adj, train_mask)
        train_accuracy = model.compute_accuracy(features, y_train, adj, train_mask)
        val_loss = model.calc_loss(features, y_val, adj, val_mask)
        val_accuracy = model.compute_accuracy(features, y_val, adj, val_mask)
        elapsed = end - start
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
          "train_acc=", "{:.5f}".format(train_accuracy), "val_loss=", "{:.5f}".format(val_loss),
          "val_acc=", "{:.5f}".format(val_accuracy), "time=", "{:.5f}".format(elapsed))

        total_time += elapsed

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                val_epoch = epoch 
            else:
                if epoch - val_epoch == 10:
                    print("validation loss has not improved for 10 epochs... stopping early")
                    break

    print("Total time: {:.4f}s".format(total_time))

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
    model.gd_update(features, y_train, adj, train_mask, lr=0.1)

def train():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
    model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])
    train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask, epochs=200)

    test_loss = model.calc_loss(features, y_test, adj, test_mask)
    test_accuracy = model.compute_accuracy(features, y_test, adj, test_mask)
    print("test_loss=", "{:.5f}".format(test_loss), "test_acc=", "{:.5f}".format(test_accuracy))

if __name__ == '__main__':
    train()
