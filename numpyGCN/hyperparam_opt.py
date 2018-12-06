from __future__ import print_function

import time
import json
from datetime import datetime
import numpy as np

from numpyGCN import numpyGCN
from utils import load_data
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#seed = 123
#np.random.seed(seed)

def train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask, early_stopping, lr, epochs):
    t_total = time.time()
    best_val_loss, val_epoch = float('inf'), 0
    past_loss = float('inf')
    for epoch in range(epochs):
        start = time.time()
        model.gd_update(features, y_train, adj, train_mask, lr)
        end = time.time()

        train_loss = model.calc_loss(features, y_train, adj, train_mask)
        val_loss = model.calc_loss(features, y_val, adj, val_mask)

        train_accuracy = model.compute_accuracy(features, y_train, adj, train_mask)
        val_accuracy = model.compute_accuracy(features, y_val, adj, val_mask)

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

def train(learning_rate, dropout, weight_decay):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
    early_stopping = True
    epochs=200
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

    val_acc = model.compute_accuracy(features, y_val, adj, val_mask)

    print("test_loss=", "{:.5f}".format(test_loss), "test_acc=", "{:.5f}".format(test_accuracy))
    return val_acc

def objective(space):
    acc = train(space['learning_rate'], space['weight_decay'], space['dropout'])
    print("learning rate " + str(space['learning_rate']) + '\n' + \
          'weight decay ' + str(space['weight_decay']) + '\n' + \
          'dropout ' + str(space['dropout']) + '\n' + \
          'overall accuracy: ' + str(acc))
    return {'loss': -acc, 'status': STATUS_OK}

if __name__ == '__main__':
    # Hyperparameter optimization
    # train(learning_rate=0.1, dropout=0.5, weight_decay=0.0005)
    #train(0.1, 0.2, 0.0005)

    space = {
             'learning_rate' : hp.uniform('learning_rate', 0.0001, 1.0), 
             'weight_decay' : hp.uniform('weight_decay', 0.0, 0.001),
             'dropout' : hp.uniform('dropout', 0.0, 1.0)
             }

    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=600)

    with open('numpyGCN_hyperparams.json', 'w') as f:
        json.dump(best, f)

