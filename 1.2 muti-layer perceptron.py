from os import X_OK
import numpy as np
import matplotlib.pyplot as plt

def muti_layer_perceptron(show_scatter_1, show_scatter_2):
    """ This function can reproduce the XOR

    Args:
        show_scatter_1 (Boolan): Plot the x graph
        show_scatter_2 (Boolan): Plot the h graph
    """
    #------------------------------------------------------------------------
    #--------------------- Define intial matrices ---------------------
    #------------------------------------------------------------------------
    X = np.array([[0,0], [0,1], [1,0], [1,1]])

    # define X_ones to be 1x4 collumn vector of ones
    X_ones = np.ones(shape=(X.shape[0], 1))
    X = np.hstack((X_ones, X))

    Y = np.array([[0], [1], [1], [0]])

    # define optimal weightings
    W = np.array([[0, -1], [1,1], [1,1]], dtype=np.float)
    w = np.array([[0], [1], [-2]], dtype=np.float)

    # define dot product
    dot_product_X_W = np.matmul(X, W)
    h = np.maximum(dot_product_X_W, 0)
    
    # repeat for h
    h_ones = np.ones(shape=(h.shape[0], 1))
    h = np.hstack((h_ones, h))

    Yhat = np.matmul(h, w)
    
    #------------------------------------------------------------------------
    #--------------------- Scatter Plot ---------------------
    #------------------------------------------------------------------------
    
    # define decision matrices
    id0 = np.where(Y[:, 0] == 0)
    id1 = np.where(Y[:, 0] == 1)

    if show_scatter_1 == True:
        plt.figure()
        plt.scatter(X[id0, 1], X[id0, 2], color='blue')
        plt.scatter(X[id1, 1], X[id1, 2], color='red')
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.savefig(r"Notes 1 Test Figures/1.2 muti layer perception model XOR for x.jpeg")
        plt.show()
    
    if show_scatter_2 == True:
        plt.figure()
        plt.scatter(h[id0, 1], h[id0, 2], color='blue')
        plt.scatter(h[id1, 1], h[id1, 2], color='red')
        plt.xlabel('$h_1$', fontsize=16)
        plt.ylabel('$h_2$', fontsize=16)
        plt.savefig(r"Notes 1 Test Figures/1.2 muti layer perception model XOR for h.jpeg")
        plt.show()
    return

muti_layer_perceptron(show_scatter_1=True, show_scatter_2=True)