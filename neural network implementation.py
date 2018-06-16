# -*- coding: utf-8 -*-


import numpy as np

"""
Based on the blog on medium at https://towardsdatascience.com/how-to-build-your own-neural-nework-from-scratch-in-python-6899a08e4f6
"""
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(y.shape)
        
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
        
    def sigmoid_derivative(self, inp):
        return inp*(1-inp)
    
    def feedforward(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2))
        
    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
#    def calculateLoss(self):
nnInput = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
nnOutput = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(nnInput, nnOutput)

for x in range(1500):
    nn.feedforward()
    nn.backprop()
    print(x)
    print(nn.output)    
