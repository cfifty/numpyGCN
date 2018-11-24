import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *

class numpyGCN:

	# two layer GCN 
	def __init__(self, input_dim, hidden_dim, output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		# randomly initialize weight matrices
		self.W_1 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
		self.W_2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, output_dim))

	# simple forward pass
	def forward(self, X, A):
		out_1 = relu((A.dot(X)).dot(self.W_1))
		return = softmax((A.dot(out_1)).dot(self.W_2))

	# argmax to predict the label
	def predict(self, x):
		return np.argmax(x, axis=1)

	# cross-entropy loss
	def calc_loss(self, preds, y):
		raise NotImplementedError

	# back propagation 
	def backprop(self, x, y):
		raise NotImplementedError

	def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
		raise NotImplementedError

	def gd_update(self, x, y, lr):
		raise NotImplementedError





