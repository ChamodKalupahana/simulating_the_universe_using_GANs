import numpy as np
import matplotlib.pyplot as plt

def back_propagtion():
    """ Let us try the XOR function using the same muti layer perceptron network 
        and using back propagation with simple gradient descent
    """

    np.random.seed(2)

    #------------------------------------------------------------------------
    #--------------------- Define inital matrices ---------------------
    #------------------------------------------------------------------------
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y = np.array([[0], [1], [1], [0]]) # Y for XOR function

    return


#------------------------------------------------------------------------
#--------------------- Define class object ---------------------
#------------------------------------------------------------------------

class MLP(object):

  def __init__(self):
    # Initialise with random weights
    self.weights_1 = 0.1 * np.random.normal(size=(3,2))
    self.weights_2 = 0.1 * np.random.normal(size=(3,1))

  def forward(self, x):
    # Do a forward pass
    if len(x.shape) == 1:
      # Single example, so add a batch dimension of 1
      x = np.expand_dims(x, axis=0)
    # Hidden layer 
    z_1 = np.matmul(np.hstack((np.ones(shape=(x.shape[0], 1)), x)), self.weights_1)
    # Apply ReLU activation function
    a_1 = np.maximum(z_1, 0)
    # Output layer
    z_2 = np.matmul(np.hstack((np.ones(shape=(a_1.shape[0], 1)), a_1)), self.weights_2)
    # Linear activation 
    a_2 = z_2
    return z_1, a_1, z_2, a_2