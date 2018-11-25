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
	def predict(self, X):
		return np.argmax(X, axis=1)

	# returns the accuracy of the classifier
	def accuracy(self, X, Y):
		idx = np.argmax(X, axis=1)
		return Y[idx].sum()/Y.shape[0]

	# calculates the unnormalized total loss with cross-entropy
	def calc_total_loss(self, X, Y, mask):
		loss = 0
		out_2 = self.forward(X)
		for idx in np.argwhere(mask == True):
			loss += np.dot(y[idx],np.log(out_2[idx]))
		return -loss 

	# normalized cross entropy loss
	def calc_loss(self, X, Y, mask):
		N = mask.sum()
		return self.calc_total_loss(X, Y, mask)/N

	# back propagation 
	def backprop(self, x, y, mask, A):
		dW_1 = np.zeros(self.W_1.shape)
		dW_2 = np.zeros(self.W_2.shape)
		
		# forward pass output
		out2 = self.forward(X)

		# last layer bp for cross entropy loss with softmax activation
		dL_dIn2 = softmax_cross_entropy_deriv(out_2, y)

		dIn2_dW2 = np.dot(A, self.out_1).T
		dL_dW2 = np.dot(dIn2_dW2, dL_dIn2)

		# next layer...
		dIn2_dOut1 = self.W_2



		# first layer bp with ReLU
		# dX/dY of X = AYW
		dLdY_1 = np.dot(self.W_2,dLdX_2)
		
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




