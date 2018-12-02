# CS 6787 Final Project: Graph Convolutional Network Analysis

In this repository we implement a Graph Convolutional Network using exclusively python primitives and compare its performance with Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)'s TensorFlow and Pytorch implementation. 

For more information on Graph Convolutional networks see [GCN](http://tkipf.github.io/graph-convolutional-networks/)

## Numpy Implementation

```bash
python numpyGCN/train.py
```

## TensorFlow Implementation

First run...
```bash
python tfGCN/setup.py install
```

and then 
```bash
python tfGCN/train.py
```

## PyTorch Implementation

```bash
python torchGCN/train.py
```