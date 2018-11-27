from datetime import datetime
import numpy as np

from utils import softmax, softmax_cross_entropy_deriv, relu, relu_diff

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
		
		# forward pass output
		out2 = self.forward(X, A)
		print("forward pass complete")

		# last layer bp for cross entropy loss with softmax activation
		dL_dIn2 = softmax_cross_entropy_deriv(out2, Y)

		print("softmax cross entropy deriv finished")
		print(A.shape)
		print(dL_dIn2.shape)
		print(self.out_1.shape)
		dIn2_dW2 = np.dot(A, self.out_1).T

		print("dIn2_dW2 finished..")
		dL_dW2 = np.dot(dIn2_dW2, dL_dIn2)

		print("dL/dW2 finished...")

		# next layer...
		dIn2_dOut1 = self.W_2
		dL_dOut1 = np.dot(dIn2_dOut1, dL_dIn2.T).T
		dIn1_dW1 = np.dot(A,X).T
		dL_dW1 = np.dot(dIn1_dW1, dL_dOut1)
		print("dL/dW1 finished...")

		return [dL_dW1, dL_dW2]


	def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
		raise NotImplementedError

	def gd_update(self, X, Y, A, mask, lr=0.1):
		# compute weight gradients
		dW_1, dW_2 = self.backprop(X,Y, A, mask)

		# parameter update
		self.W_1 -= dW_1*lr
		self.W_2 -= dW_2*lr 




