import numpy as np

def softmax(X):
	return np.exp(X) / np.sum(np.exp(X), axis=1)[:,None]

def relu(X):
	zeros = np.maximum(X, np.zeros((X.shape)))