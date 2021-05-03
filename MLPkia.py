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
import matplotlib.pyplot as plt

import math


# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        self.W = W
        self.b = b
    # DEFINE __init function

    def forward(self, x):
    # DEFINE forward function
       self.x = x
    #    print("self w in forward")
    #    print(self.W.shape)
       self.product = np.dot(self.x, self.W) + self.b

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
# This layer will be used twice:
        # first: it'll receive dL/dh2 signal to correct params on LT output
        # dL/dweights_l2 = dL/dh2 * dh2/dweights_l2 = dL/dh2 * z1
        # dL/db2 = dL/dh2 * 1

        # second: it'll receive dL/dh1 signal to correct LT hidden layer
        # dL/dw1 = dL/dh1 * dh1/dw1 = dL/dh1 * x
        # dL/dbiases_l1 = dL/dh2 * 1

        # finally: when at LT output layer, this should supply dL_dz1
        # as an input to ReLU layer
        # Let der_sigma(h2) = K
        # dL/dz1 = dL/dz2 * dz2/dz1 = K*(z2 - y')*weights_l2 / K = (z2 - y') * weights_l2
     
     dx =  np.dot(grad_output, self.W.T)
     dW = np.dot(self.x.T, grad_output)
     db = np.sum(grad_output, axis = 0)
     return dx, dW, db









# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def __init__(self):
       self.x = None
       

    def forward(self, x):
    # DEFINE forward function
      self.x = x
      self.x = np.maximum(0, self.x)
      return self.x


#   
    @staticmethod
    def ReLU_derivative(value):
        value[value > 0] = 1
        return value

    def backward(
        self, 
        grad_output,
        
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
    # DEFINE backward function
     self.x[self.x > 0] = 1
     dr = np.multiply(self.x, grad_output)
     return dr
# ADD other operations in ReLU if needed






# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):


    def __init__(self):
        self.y = None
        self.output = None
        # self.dir = direction
        # if self.dir == 'forward':
        #     self.forward(self.x)
        # elif self.dir == 'backward':
        #     self.backward()

    
    @staticmethod
    def sigmoid(x):
        #expit(x)

        return 1.0/(1.0 + np.exp(-x) )


    def forward(self,x):
        # DEFINE forward function\ 
            self.x = x
            self.output = self.sigmoid(self.x)
            return self.output



    def backward(
        self, 
        y,
        grad_output, 
       
        learning_rate=0.0,
        momentum=0.0,
        l2_penalty=0.0,
        
    ):
         self.y = y
         
         dE_dh2  = grad_output * (self.output - self.y)
         return dE_dh2
        # DEFINE backward function
# ADD other operations and data entries in SigmoidCrossEntropy if needed








# This is a class for the Multilayer perceptron
class MLP(object):

    def __init__(self, input_dims, hidden_units, labels, batch_size,output_dim = None):

    # INSERT CODE for initializing the network
        if output_dim != None:
         self.output_dimensions = output_dim
        self.b_size = batch_size
        self.class_labels = labels
        self.in_dim = input_dims
        self.hidden = hidden_units
        self.batch_loss = []
        self.batch_acc = []
        self.test_losses = []
        self.test_accuracies = []
       


        # Kiaming initialization of weights and Initialization of biases 
        self.weights_l1 = np.random.uniform(-1.0, 1.0, size=(self.in_dim, self.hidden)) * math.sqrt(2/self.in_dim)
        self.biases_l1 = np.random.uniform(-1.0, 1.0, size=(1, self.hidden)) 
        self.weights_l2 = np.random.uniform(-1.0, 1.0, size=(self.hidden, self.output_dimensions)) * math.sqrt(2/self.hidden)
        self.biases_l2 = np.random.uniform(-1.0, 1.0, size=(1, self.output_dimensions))



        # Instantiation of the classes
        self.layer1 = LinearTransform(self.weights_l1, self.biases_l1)
        self.layer2 = LinearTransform(self.weights_l2, self.biases_l2)
        self.relu = ReLU()
        self.sigcr = SigmoidCrossEntropy()



        #Momentum initialization
        self.weights_l1_momentum = 0
        self.biases_l1_momentum = 0
        self.weights_l2_momentum = 0
        self.biases_l2_momentum = 0
        

    def forward_pass(self, x):
        self.x = x
        self.layer1_forward = self.layer1.forward(self.x)
        self.activation1 = self.relu.forward(self.layer1_forward)
        self.layer2_forward = self.layer2.forward(self.activation1)
        self.pred = self.sigcr.forward(self.layer2_forward)
        return self.pred



    def backward_pass(self, y, lr, momentum, penality):
        self.y_labels = y
       
        
        defaults = {
                "learning_rate": 0.0,
                "momentum": 0.0,
                "l2_penalty": 0.0
            }

        self.pred_back = self.sigcr.backward(self.y_labels, 1, **defaults)
        dx2, dw2, db2 = self.layer2.backward(self.pred_back, **defaults)
        self.activation1_back = self.relu.backward(dx2, **defaults)    
        dx1, dw1, db1 = self.layer1.backward(self.activation1_back, **defaults)
        

        # Mini-batch gradient descent
        dw1 = dw1/self.b_size
        db1 = db1/self.b_size
        dw2 = dw2/self.b_size
        db2 = db2/self.b_size

        # Updating weights and biases.
        self.weights_l1_momentum = momentum * self.weights_l1_momentum - lr * dw1
        self.biases_l1_momentum = momentum * self.biases_l1_momentum - lr * db1
        self.weights_l2_momentum = momentum * self.weights_l2_momentum - lr * dw2
        self.biases_l2_momentum = momentum * self.biases_l2_momentum - lr * db2

        self.weights_l1 += self.weights_l1_momentum
        self.biases_l1 += self.biases_l1_momentum
        self.weights_l2 += self.weights_l2_momentum
        self.biases_l2 += self.biases_l2_momentum
        
        #loss and accuracy calculations
        
        loss = -1 * (y_batch * np.log(self.pred + 10e-8) + (1.0 - y_batch) * np.log(1.0 - self.pred + 10e-8))
        loss += penality / 2 * (np.sum(np.square(self.weights_l1)) + np.sum(np.square(self.weights_l2)))

        mean_loss = np.mean(loss)

        batch_accuracy = 100 * np.mean((np.round(self.pred) == y_batch))

        self.batch_loss.append(mean_loss)
        self.batch_acc.append(batch_accuracy)

        # self.layer1.update_weights(0.0001, 0.6)
        # self.layer2.update_weights(0.0001, 0.6)
        return mean_loss, batch_accuracy


           
    # TRAIN Function
    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate =10e-3, 
        momentum = 0.9,
        l2_penalty = 0.0,
        
    ):
    # INSERT CODE for training the network
      
        self.forward_pass(x=x_batch)
        loss, training_acc = self.backward_pass(y=y_batch, lr = learning_rate, momentum = momentum, penality = l2_penalty)
        self.train_labels = y_batch
      
        return loss, training_acc
       # return loss





    # TEST Function
    def evaluate(self,x_test, y_test, l2_penalty):
    # INSERT CODE for testing the network
    # ADD other operations and data entries in MLP if needed
       
        y_prob = self.forward_pass(x=x_test)
        y_pred = np.round(y_prob)
        
        test_loss = -1 * (y_test * np.log(y_prob + 10e-8) + (1.0 - y_test) * np.log(1.0 - y_prob + 10e-8))
        test_loss += l2_penalty / 2 * (np.sum(np.square(self.weights_l1)) + np.sum(np.square(self.weights_l2)))

        self.test_losses.append(np.mean(test_loss))
        self.test_accuracies.append(100 * np.mean(y_pred == y_test))


        return np.mean(self.test_losses), np.mean(self.test_accuracies)


        






if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else :
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')

    train_x = data['train_data']
    train_y = data['train_labels']
    test_x = data['test_data']
    test_y = data['test_labels']

   
    num_examples, input_dims = train_x.shape

    # data normalization
    x_max_train = np.max(train_x, axis=0)
    train_x = train_x / x_max_train
    x_max_test = np.max(test_x, axis=0)
    test_x = test_x / x_max_test

    x_mean_train = np.mean(train_x, axis=0)
    train_x = (train_x - x_mean_train)
    x_mean_test = np.mean(test_x, axis=0)
    test_x = (test_x - x_mean_test)






    
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 30
    test_accs=[]
    training_losses = []
    training_accuracies = []
    test_losses = []
    num_batches = 100
    hidden_units = 2046
    batch_size = num_examples // num_batches
    print(batch_size)
    
    output_d = 1
    mlp = MLP(input_dims, hidden_units, train_y, batch_size,output_d)

    for epoch in range(num_epochs):

    # INSERT YOUR CODE FOR EACH EPOCH HERE

        for b in range(num_batches):
            total_loss = 0.0
            x_batch = train_x[b * batch_size: (b + 1) * batch_size, :]
            y_batch = train_y[b * batch_size: (b + 1) * batch_size, :]
            
            total_loss, train_acc= mlp.train(x_batch, y_batch)
            training_losses.append(np.mean(total_loss))
            training_accuracies.append(np.mean(train_acc))

            

            #MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
            sys.stdout.flush()
            print(total_loss)    
        # # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        test_loss, test_accuracy= mlp.evaluate(
                x_test=test_x, y_test=test_y, l2_penalty=0.0
            )
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)    

     
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
 
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            round(np.mean(training_losses), 3),
            round(np.mean(training_accuracies), 3)
        ))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            round(np.mean(test_losses), 3),
            round(np.mean(test_accs), 3)
        )
        )

    plt.plot(test_accs)
    plt.savefig("test_acc_hu_2046.jpg")
    plt.show()    