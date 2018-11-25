import operator
import numpy as np
import sys
from datetime import datetime
from utils import *

class numpyGCN:

	# two layer GCN 
	def __init__(self, input_dim, hidden_dim, output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.out_1 = None

		# randomly initialize weight matrices
		self.W_1 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
		self.W_2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, output_dim))

	# simple forward pass
	def forward(self, X, A):
		self.out_1 = relu((A.dot(X)).dot(self.W_1))
		return softmax((A.dot(self.out_1)).dot(self.W_2))

	# argmax to predict the label
	def predict(self, x):
		return np.argmax(x, axis=1)

	# cross-entropy loss
	def calc_loss(self, mask, out_2, y):
		loss = 0
		for idx in np.argwhere(mask == True):
			loss += np.dot(y[idx],np.log(out_2[idx]))
		return -loss 

	# back propagation 
	def backprop(self, x, y, mask, A):
		dW_1 = np.zeros(self.W_1.shape)
		dW_2 = np.zeros(self.W_2.shape)
		
		# forward pass output
		out_2 = self.forward(X)

		# last layer bp for cross entropy loss with softmax activation
		dLdX_2 = out_2[mask] - y[mask]

		# TODO: fix dX_2dW_2 when I remember linear algebra derivatives...
		# dX/dW of X = AYW 
		dX_2dW_2 = np.dot(A, self.out_1)


		dLdW_2 = np.dot(dX_2dW_2.T, dLdX_2)

		# first layer bp with ReLU
		# TODO: fix when I remember lienar algebra derivatives...
		# dX/dY of X = AYW
		dLdY_1 = np.dot(,dLdX_2)
		
		dLdX_1 = relu_diff(A.dot(X).dot(self.W_1)).dot(dLdY_1)
		
		# TODO; fix once we can comput gradent w.r.t. a symmetric matrix
		dX_1dW_1 = np.dot(A, X)

		dLdW_1 = dX_1dW_1.dot(dLdX_1)

		return [dLdW_1, dLdW_2]


	def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
		raise NotImplementedError

	def gd_update(self, X, Y, lr):
		# compute weight gradients
		dW_1, dW_2 = self.backprop(X,Y)

		# parameter update
		self.W_1 -= dW_1*lr
		self.W_2 -= dW_2*lr 




