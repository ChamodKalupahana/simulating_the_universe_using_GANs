from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def perceptron_model_and(show_scatter_1):
    """ Let's attempt to learn the AND function
    """ 
    #------------------------------------------------------------------------
    #--------------------- Define matrices ---------------------
    #------------------------------------------------------------------------
    X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    Y = np.array([[0], [0], [0], [1]])

    np.random.seed(0)   
    
    # np.random.random returns random floats between 0 and 1
    w = 0.1 * np.random.random(size=(3, 1))

    #------------------------------------------------------------------------
    #--------------------- Iteration Method for w ---------------------
    #------------------------------------------------------------------------

    num_epochs = 10
    learning_rate = 0.1

    for i in range(num_epochs):
        dot_product_X_w = np.matmul(X, w)
        
        # np.heaviside returns 0 if dot_product_X_w is less than 0
        # np.heaviside returns the second input (in this case 0 too) if dot_product_X_w is equal to 0
        # np.heaviside returns 1 if dot_product_X_w is greater than 0

        # Yhat changes on each iteration too
        Yhat = np.heaviside(dot_product_X_w, 0)
        dot_product_X_Yhat = np.matmul(X.T, Yhat - Y)

        # w is put through 10 iterations and w settles 
        w -= learning_rate * dot_product_X_Yhat

    #------------------------------------------------------------------------
    #--------------------- Show correct points for AND function ---------------------
    #------------------------------------------------------------------------

    # define decision matrices
    id0 = np.where(Y[:, 0] == 0)
    id1 = np.where(Y[:, 0] == 1)

    # define xx to be a 300x300 array which varies from -1 to 2 in steps of 0.01 left to right
    # define yy to be a 300x300 array which varies from -1 to 2 in steps of 0.01 top to bottom?
    xx, yy = np.mgrid[-1:2:0.01, -1:2:0.01]
    
    # define new Yhat
    w_array = w[0] + w[1] * xx + w[2] * yy
    Yhat = np.heaviside(w_array, 0)

    if show_scatter_1 == True:
        plt.figure()
        
        #plt.contourf plots filled coutours
        plt.contourf(xx, yy, Yhat, alpha=0.5)
        plt.scatter(X[id0, 1], X[id0, 2], color='blue')
        plt.scatter(X[id1, 1], X[id1, 2], color='red')
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.savefig(r"Notes 1 Test Figures/1.2 perception model AND.jpeg")
        plt.show()

        # the perceptron model classifies the and function correctly because only the (x_1 = 1, x_2 = 1) is red 
        # and the other points are blue

    return

def perceptron_model_xor(show_scatter_1):
    """ Let's attempt to learn the XOR function
    """ 
    #------------------------------------------------------------------------
    #--------------------- Define matrices ---------------------
    #------------------------------------------------------------------------
    
    X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    Y = np.array([[0], [1], [1], [0]])

    np.random.seed(0)
    w = 0.1 * np.random.random(size=(3, 1))
    
    #------------------------------------------------------------------------
    #--------------------- Iteration Method for w ---------------------
    #------------------------------------------------------------------------

    num_epochs = 10
    learning_rate = 0.1

    for i in range(num_epochs):
        Yhat = np.heaviside(np.matmul(X, w), 0)
        w -= learning_rate * np.matmul(X.T, Yhat - Y)

    #------------------------------------------------------------------------
    #--------- Show correct points for XOR function --------
    #------------------------------------------------------------------------
    Yhat = np.heaviside(np.matmul(X, w), 0)
    
    id0 = np.where(Y[:, 0] == 0)
    id1 = np.where(Y[:, 0] == 1)

    xx, yy = np.mgrid[-1:2:0.01, -1:2:0.01]
    Yhat = np.heaviside(w[0] + w[1] * xx + w[2] * yy, 0)

    if show_scatter_1 == True:
        plt.figure()
        plt.contourf(xx, yy, Yhat, alpha=0.5)
        plt.scatter(X[id0, 1], X[id0, 2], color='blue')
        plt.scatter(X[id1, 1], X[id1, 2], color='red')
        plt.xlabel('$x_1$', fontsize=16)
        plt.ylabel('$x_2$', fontsize=16)
        plt.savefig(r'Notes 1 Test Figures\1.2 perception model XOR.jpeg')
        plt.show()

        # the perceptron model doesn't predict the XOR function correctly because the red points (x1=0, x2=1)
        # and (x2=1, x1=0) should be shaded but the prediction fails here
    return



#perceptron_model_and(show_scatter_1=True)
perceptron_model_xor(show_scatter_1=True)