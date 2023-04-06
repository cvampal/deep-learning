#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.random.seed(42)

def xavier_init(shape):
    stddev = np.sqrt(2.0 / (shape[0] + shape[1]))
    return np.random.normal(0, stddev, shape)

sigmoid = lambda x : 1 / (np.exp(-x) + 1) 

def forward_and_backward(x,y,parameters,do_backward=True):
    w0,b0,w1,b1,w2,b2 = parameters
    dparameters = []
    # layer I
    x0 = (x @  w0) + b0
    x0sig = sigmoid(x0)
    # layer II
    x1 = (x0sig @ w1) + b1
    x2 = sigmoid(x1)
    # layer III
    logit = (x2 @ w2) + b2

    # softmax of output
    logit_max = logit.max(1, keepdims=True)
    norm_logits = logit - logit_max  # for numerical stability while calculating softmax
    counts = np.exp(norm_logits)
    counts_sum = counts.sum(1, keepdims=True)
    probs = counts*counts_sum**(-1)

    # cross entropy loss 
    logprobs = np.log(probs)
    loss = -logprobs[range(x.shape[0]), y].mean()

    if do_backward:
    # backward pass
        dlogprobs = np.zeros_like(logprobs) 
        dlogprobs[range(batch_size), y] = -1/batch_size
        dprobs =  probs**(-1) * dlogprobs
        dcounts = counts_sum**(-1)  * dprobs
        dcounts_sum = (-1* counts * counts_sum**(-2) * dprobs).sum(1, keepdims=True)
        dcounts += np.ones_like(counts) * dcounts_sum
        dnorm_logits =  counts * dcounts
        dlogits =  dnorm_logits.copy()
        dlogit_max = (-1* dnorm_logits).sum(1, keepdims=True)
        tmp = np.zeros_like(logit)
        tmp[range(logit.shape[0]),logit.argmax(1, keepdims=True) ] = 1
        dlogits += tmp *dlogit_max
        dx2 = dlogits @ w2.T  
        dw2 = x2.T @ dlogits
        db2 = dlogits.sum(0)
        dx1 = x2 * (1-x2) * dx2
        dx0sig = dx1 @ w1.T
        dw1 = x0sig.T @ dx1
        db1 = dx1.sum(0)
        dx0 = x0sig *(1-x0sig) * dx0sig
        dx = dx0 @ w0.T
        dw0 = x.T @ dx0
        db0 = dx0.sum(0)
        dparameters = [dw0,db0,dw1,db1,dw2,db2]
    return probs, loss, dparameters

def update_parameters(parameters, dparameters, lr):
    for i in range(len(parameters)):
        parameters[i] -= lr * dparameters[i]
    return parameters

def get_accuracy(y_pred_probs, y_true):
    return (np.array(y_pred_probs.argmax(1)) == y_true.ravel()).sum() / y_pred_probs.shape[0]

def evaluate(x,y,parameters):
    probs, _ , _ = forward_and_backward(x,y,parameters,do_backward=False)
    acc = get_accuracy(probs, y)
    return acc

import sklearn.datasets
import sklearn.model_selection
from tqdm import tqdm

## dataset
X,Y = sklearn.datasets.fetch_olivetti_faces(shuffle=True, random_state=42, return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33, random_state=42)

batch_size = 32
inp = X_train.shape[-1]
h0 = 512
h1 = 128
h2 = 40
lr = 0.01

# weights of the model
w0 = xavier_init((inp,h0))
b0 = np.zeros(h0)
w1 = xavier_init((h0,h1)) 
b1 = np.zeros(h1)
w2 = xavier_init((h1,h2))
b2 = np.zeros(h2)
parameters = [w0,b0,w1,b1,w2,b2]

train_loss = []
total_data = X_train.shape[0]
num_epochs = 7000

# training loop
print("Training:")
for i in tqdm(range(num_epochs)):
    rand_idx = np.random.randint(0,total_data,(batch_size,))
    x = X_train[rand_idx]
    y = y_train[rand_idx]
    _, loss, dpara = forward_and_backward(x,y, parameters)
    parameters = update_parameters(parameters, dpara, lr=lr)
    train_loss += [loss]

import matplotlib.pyplot as plt
#plot train loss
plt.plot(np.array(train_loss).reshape(-1,100).mean(1))
plt.show()
# accuracy on test
print(f"Test accuracy: {evaluate(X_test, y_test, parameters) :.4f}")
