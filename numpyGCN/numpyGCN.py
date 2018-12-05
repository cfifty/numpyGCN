from datetime import datetime
import numpy as np

from utils import softmax, softmax_cross_entropy_deriv, relu, relu_diff

class numpyGCN:

    # two layer GCN
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=None, weight_decay=0, random_noise=False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.in_1 = None
        self.out_1 = None
        self.in_2 = None
        self.out_2 = None
        self.random_noise = True

        self.dropout = dropout
        self.weight_decay = weight_decay

        # randomly initialize weight matrices according to Glorot & Bengio (2010)
        self.W_1 = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (input_dim, hidden_dim))
        self.W_2 = np.random.uniform(-np.sqrt(1./output_dim), np.sqrt(1./output_dim), (hidden_dim, output_dim))

    # simple forward pass
    def forward(self, X, A, drop_weights=None):
        W_1 = self.W_1
        W_2 = self.W_2

        if drop_weights:
            d1, d2 = drop_weights
            W_1 = W_1 * d1
            #W_2 = W_2 * d2

        self.in_1 = A.dot(X).dot(W_1)
        self.out_1 = relu(self.in_1)
        self.in_2 = A.dot(self.out_1).dot(W_2)
        self.out_2 = softmax(self.in_2)
        return self.out_2

    # argmax to predict the label
    def predict(self, X, A):
        out = self.forward(X, A)
        return np.argmax(out, axis=1)

    # returns the accuracy of the classifier
    def compute_accuracy(self, X, Y, A, mask):
        out = self.forward(X, A)
        out_class = np.argmax(out[mask], axis=1)
        expected_class = np.argmax(Y[mask], axis=1)
        num_correct = np.sum(out_class == expected_class).astype(float)
        return num_correct / expected_class.shape[0]

    # normalized cross entropy loss
    def calc_loss(self, X, Y, A, mask):
        N = mask.sum()
        preds = self.forward(X, A)
        loss = np.sum(Y[mask] * np.log(preds[mask]))
        loss = np.asscalar(-loss) / N

        if self.weight_decay:
            l2_reg = np.sum(np.square(self.W_1)) * self.weight_decay / 2
            loss = loss + l2_reg

        return loss

    # back propagation
    def backprop(self, X, Y, A, mask):
        dW_1 = np.zeros(self.W_1.shape)
        dW_2 = np.zeros(self.W_2.shape)

        if self.random_noise:
            tmp_W1, tmp_W2 = self.W_1, self.W_2
            self.W_1 += np.random.normal(0, 0.001, self.W_1.shape)
            self.W_2 += np.random.normal(0, 0.001, self.W_2.shape)

        # divide by d so expectation of GCN layer doesn't change from train to test
        if self.dropout:
            d1 = np.random.binomial(1, (1-self.dropout), size=self.W_1.shape) / (1-self.dropout)
            d2 = np.random.binomial(1, (1-self.dropout), size=self.W_2.shape) / (1-self.dropout)
            preds = self.forward(X, A, (d1,d2))
        else:
            preds = self.forward(X, A)

        # IMPORTANT: update gradient based only on masked labels
        preds[~mask] = Y[~mask]

        # last layer bp for cross entropy loss with softmax activation
        dL_dIn2 = softmax_cross_entropy_deriv(preds, Y)

        dIn2_dW2 = A.dot(self.out_1).transpose()
        dL_dW2 = dIn2_dW2.dot(dL_dIn2)

        # apply backprop for next layer
        dL_dOut1 = A.transpose().dot(dL_dIn2).dot(self.W_2.transpose())

        dOut1_dIn1 = relu_diff(self.in_1)
        dL_dIn1 = dL_dOut1 * dOut1_dIn1
        dIn1_dW1 = A.dot(X).transpose()

        dL_dW1 = dIn1_dW1.dot(dL_dIn1)

        if self.weight_decay:
            dL_dW1 += self.weight_decay * self.W_1

        if self.dropout:
            dL_dW1 *= d1
            #dL_dW2 *= d2

        if self.random_noise:
            self.W_1, self.W_2 = tmp_W1, tmp_W2

        return (dL_dW1, dL_dW2)

    def gd_update(self, X, Y, A, mask, lr):
        # compute weight gradients
        dW_1, dW_2 = self.backprop(X, Y, A, mask)

        # parameter update
        self.W_1 -= dW_1 * lr
        self.W_2 -= dW_2 * lr