"""
Nandam Sri ranga Chaitanya
"""


from __future__ import division
from __future__ import print_function

import sys
try:
   import _pickle as pickle
except:
   import pickle

import numpy as np
from collections import OrderedDict
import decimal
from scipy.special import expit

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __call__(self,x):
        self.initParams()


    def __init__(self, input_dims, neuron_count):
        self.columns = neuron_count
       
        self.rows = input_dims
        self. matrix_dims = [self.rows, self.columns]
        self.W = np.random.randn(*self.matrix_dims)
        self.b = np.random.randn(self.columns)
        self.mW = 0
        self.mb = 0

        
    # DEFINE __init function

    def forward(self, x):
    # DEFINE forward function
       self.x = x
       self.product = np.matmul(self.x, self.W) + self.b

       return self.product


    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
# ADD other operations in LinearTransform if needed
     self.dW = np.matmul(self.x.T, grad_output)
     self.db = np.sum(grad_output, axis = 0)
     

     self.dW += momentum * self.mW - learning_rate * self.dW
     self.db += momentum * self.md - learning_rate * self.db


     return np.matmul(grad_output, self.W.T)









# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def __init__(self,x):
        self.x = x
        # self.dir = direction
        # if self.dir == 'forward':
        #     self.forward(self.x)
        # elif self.dir == 'backward':
        #     self.ReLU_derivative(self.x)    
        

    def forward(self, x):
    # DEFINE forward function
      self.x[self.x <= 0] = 0

      return self.x


#    This has to be changed!!   PLAGIARISM!!!
    @staticmethod
    def ReLU_derivative(value):
        value[value > 0] = 1
        value[value <= 0] = 0


        return value

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
     return self.ReLU_derivative(self.product) * grad_output 
# ADD other operations in ReLU if needed






# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):


    def __init__(self, h2):
        self.x = h2
        # self.dir = direction
        # if self.dir == 'forward':
        #     self.forward(self.x)
        # elif self.dir == 'backward':
        #     self.backward()

    
    @staticmethod
    def sigmoid(x):
        expit(x)
        return 1/(1 + np.exp(-x) )


    def forward(self, x):
        # DEFINE forward function\ 
            self.output = self.sigmoid(self.x)
            return self.output





    def backward(
        self, 
        grad_output, 
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0
    ):
         dL_dh2  = self.out - grad_output
         return dL_dh2
        # DEFINE backward function
# ADD other operations and data entries in SigmoidCrossEntropy if needed








# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units):

    # INSERT CODE for initializing the network
        self.in_dim = input_dims
        self.hidden = hidden_units

    

    def forward_pass(self, x, output_dim):
        self.x = x
        self.output_dimensions = output_dim
        self.layer1 = LinearTransform(self.in_dim, self.hidden)
        self.layer1 = self.layer1.forward(self.x)
        self.activation1 = ReLU(self.layer1)
        self.activation1 = self.activation1.forward(self.layer1)
        self.output_layer = LinearTransform(self.hidden, self.output_dimensions)
        self.output_layer = self.output_layer.forward(self.activation1)
        self.pred = SigmoidCrossEntropy(self.output_layer)
        self.pred = self.pred.forward(self.output_layer)

        return self.pred
    

    def binary_cross_entropy(class_probs, class_labels):

        def cross_entropy(y, p):
            return y * np.log(p) + (1 - y) * np.log(1 - p)

        return -1.0
        return -np.sum([
            cross_entropy(y=class_labels[i], p=class_probs[i])
            for i in range(len(class_probs))
        ])
      
    def backward_pass(self, y):
        self.y_labels = y
        self.sequential(stack=self.nn_stack, input=y, direction="reversed")
        return 0, 0                

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
        output_dim,
    ):
    # INSERT CODE for training the network
        self.output = output_dim
        self.forward_pass(x=x_batch, output_dim = self.output)
        #self.backward_pass(y=y_batch)
        #loss = self.loss_with_l2(l2_penalty=l2_penalty)
        loss = 1
        return self.pred
       # return loss

    def evaluate(self, x, y):
    # INSERT CODE for testing the network
# ADD other operations and data entries in MLP if needed

        train_loss = np.mean(self.loss_history)
        y_prob = self.forward_pass(x=x_train)
        train_accuracy = self.accuracy(y_prob, y_train)

        y_prob = self.forward_pass(x=x_test)
        test_loss = self.loss_with_l2(l2_penalty=0.)
        test_accuracy = self.accuracy(y=y_prob, y_labels=y_test)

        return train_loss, train_accuracy, test_loss, test_accuracy

    @staticmethod
    def accuracy(y, y_labels):
        return np.sum(y.round() == y_labels) / len(y)






if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else :
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

    train_x = train_x / np.max(train_x, axis=0)
    train_x = (train_x - np.mean(train_x, axis=0))

    test_x = test_x / np.max(test_x, axis=0)
    test_x = (test_x - np.mean(test_x, axis=0))
    
    num_examples, input_dims = train_x.shape
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 10
    num_batches = 1000
    hidden_units = 5
    batch_size = num_examples // num_batches
    
    output_d = 1
    mlp = MLP(input_dims, hidden_units)

    for epoch in range(num_epochs):

    # INSERT YOUR CODE FOR EACH EPOCH HERE

        for b in range(num_batches):
            total_loss = 0.0
            x_batch = train_x[b * batch_size: (b + 1) * batch_size, :]
            y_batch = train_y[b * batch_size: (b + 1) * batch_size, :]
            total_loss += mlp.train(x_batch, y_batch, 0.1, 0.8, 0.5, output_d)
            print(total_loss)

            # MAKE SURE TO UPDATE total_loss
            # print(
            #     '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
            #         epoch + 1,
            #         b + 1,
            #         total_loss,
            #     ),
            #     end='',
            # )
            sys.stdout.flush()
        # # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # train_loss, train_accuracy, test_loss, test_accuracy = mlp.evaluate(
        #     x_test=test_x, y_test=test_y, x_train=train_x, y_train=train_y
        # )
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
 
        # print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
        #     train_loss,
        #     100. * train_accuracy
        # ))
        # print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
        #     test_loss,
        #     100. * test_accuracy
        # )
        # )