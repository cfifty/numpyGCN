from datetime import datetime
import numpy as np
import scipy.sparse as sp

from numpyGCN import numpyGCN
from utils import load_data


# coordinate list sparse matrix for normalized adjacency matrix
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask, lr=0.005, epochs=100):
	losses = []
	for epoch in range(epochs):
		train_loss = model.calc_loss(features, y_train, adj, train_mask)
		val_loss = model.calc_loss(features, y_val, adj, val_mask)
		time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		print("%s: Train/Valid Loss after epoch=%d: %f :: %f" % (time, epoch, train_loss, val_loss))
		model.gd_update(features, y_train, adj, train_mask, lr=0.1)

# test the forward pass
def test_forward():
	adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
	adj = normalize_adj(adj)
	model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])
	
	preds = model.predict(features, adj)

	print("preds.shape " + str(preds.shape))
	print("accuracy: " + str(model.compute_accuracy(features, y_test, adj, test_mask)))

def test_calc_loss():
	adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
	adj = normalize_adj(adj)
	model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])

	loss = model.calc_loss(features, y_train, adj, train_mask)
	print("cross entropy loss " + str(loss))

# test one gradient descent update
def test_gd_step():
	adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
	adj = normalize_adj(adj)
	model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])
	model.gd_update(features, y_train, adj, train_mask, lr=0.1)

def run_model():
	adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('Cora')
	adj = normalize_adj(adj)
	model = numpyGCN(input_dim=features.shape[1], hidden_dim=16, output_dim=y_train.shape[1])
	train_with_gd(model, features, adj, y_train, y_val, train_mask, val_mask)

	test_loss = model.calc_loss(features, y_test, test_mask)
	print("Test Loss : %f" % test_loss)


if __name__ == '__main__':
	run_model()
