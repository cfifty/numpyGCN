from datetime import datetime
import numpy as np

from utils import softmax, softmax_cross_entropy_deriv, relu, relu_diff

class numpyGCN:

	# two layer GCN 
	def __init__(self, input_dim, hidden_dim, output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.in_1 = None
		self.out_1 = None
		self.in_2 = None
		self.out_2 = None

		# randomly initialize weight matrices
		self.W_1 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
		self.W_2 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, output_dim))

	# simple forward pass
	def forward(self, X, A):
		self.in_1 = A.dot(X).dot(self.W_1)
		self.out_1 = relu(self.in_1)
		self.in_2 = A.dot(self.out_1).dot(self.W_2)
		self.out_2 = softmax(self.in_2)
		return self.out_2

	# argmax to predict the label
	def predict(self, X, A):
		out = self.forward(X, A)
		return np.argmax(out, axis=1)

	# returns the accuracy of the classifier
	def compute_accuracy(self, X, Y, A, mask):
		out = self.forward(X, A)
		Y = Y[mask]
		out = out[mask]
		idx = np.argmax(out, axis=1)
		
		num_correct = 0
		for i in range(Y.shape[0]):
			num_correct += Y[i, idx[i]]
		return num_correct/Y.shape[0]

	# calculates the unnormalized total loss with cross-entropy
	def calc_total_loss(self, X, Y, A, mask):
		loss = 0
		out_2 = self.forward(X, A)
		for idx in np.argwhere(mask == True):
			loss += np.dot(Y[idx],np.log(out_2[idx].T))
		return -loss 

	# normalized cross entropy loss
	def calc_loss(self, X, Y, A, mask):
		N = mask.sum()
		return (self.calc_total_loss(X, Y, A, mask)/N)[0][0]

	# back propagation 
	def backprop(self, X, Y, A, mask):
		dW_1 = np.zeros(self.W_1.shape)
		dW_2 = np.zeros(self.W_2.shape)

		print("w1", self.W_1.shape)
		print("w2", self.W_2.shape)

		# forward pass output
		preds = self.forward(X, A)
		print("forward pass complete")

		# last layer bp for cross entropy loss with softmax activation
		dL_dIn2 = softmax_cross_entropy_deriv(preds, Y)
		print("dL_dIn2", dL_dIn2.shape)

		print("softmax cross entropy deriv finished")
		print("out1", self.out_1.shape)
		print("A", A.shape)
		print("preds", preds.shape)
		dIn2_dW2 = A.dot(self.out_1)
		print("dIn2_dW2 finished..")

		dL_dW2 = dIn2_dW2.transpose().dot(dL_dIn2)
		print("dl_dw2", dL_dW2.shape)
		print("dL/dW2 finished...")

		# next layer...
		# TODO: figure out how to compute dIn2/dOut1
		dIn2_dOut1 = self.in_2.transpose() # dL_dW2.transpose().dot(self.out_1.transpose()).dot(A.transpose())
		print("dIn2_dOut1", dIn2_dOut1.shape)

		print("dIn2_dOut1 finished...")

		print("in2", self.in_2.shape)
		print("in1", self.in_1.shape)
		print("X", X.shape)

		dL_dOut1 = dL_dIn2.dot(dIn2_dOut1)
		print("dL_dOut1", dL_dOut1.shape)

		print("dL/dOut1 finished...")

		dOut1_dIn1 = relu_diff(self.in_1)
		dL_dIn1 = dL_dOut1.dot(dOut1_dIn1)
		print("dl_dIn1", dL_dIn1.shape)

		print("dL/dIn1 finished...")

		dIn1_dW1 = A.dot(X)
		dL_dW1 = dIn1_dW1.transpose().dot(dL_dIn1)
		print("dL_dW1", dL_dW1.shape)
		print("dL/dW1 finished...")

		return (dL_dW1, dL_dW2)


	def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
		raise NotImplementedError

	def gd_update(self, X, Y, A, mask, lr=0.1):
		# compute weight gradients
		dW_1, dW_2 = self.backprop(X, Y, A, mask)

		# parameter update
		self.W_1 -= dW_1 * lr
		self.W_2 -= dW_2 * lr
