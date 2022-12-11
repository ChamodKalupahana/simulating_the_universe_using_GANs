from operator import matmul
import numpy as np
import matplotlib.pyplot as plt
#work until half 5 for both this and imaging

def linear_model_and(show_scatter_1, show_scatter_2, show_scatter_3):
    #------------------------------------------------------------------------
    #--------------------- Define AND gate ---------------------
    #------------------------------------------------------------------------

    # what is X and Y??
    X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])

    # np.maxtrix is the matrix product for 3x3 arrays in this case:
    # X.t is the transpose of array X
    dot_product_X = np.matmul(X.T, X)

    # np.linalg.inv take the inverse of the matrix dot_product_X
    inv = np.linalg.inv(dot_product_X)

    Y = np.array([[0], [0], [0], [1]])

    if show_scatter_1 == True:
        # show y = 0 points in black and y = 1 points in red????
        plt.scatter(X[:,1], X[:,2],c=np.reshape(Y,4), edgecolors='black', cmap='gray');
        plt.show()

    # define weights??
    dot_product_X_Y = np.matmul(X.T, Y)
    w = np.matmul(inv,  dot_product_X_Y)

    Yhat = np.matmul(X, w)

    if show_scatter_2 == True:
        plt.scatter(X[:,1], X[:,2],c=np.reshape(Yhat,4), edgecolors='black', cmap='gray');
        plt.show()

    #------------------------------------------------------------------------
    #--------------------- Model Training ---------------------
    #------------------------------------------------------------------------
    
    # if we have new prediction points, we can use a trained model to predict the Y values
    n = 100
    
    # creates 200 random numbers between 0 and 1
    random_num = np.random.uniform(0, 1, n*2)
    
    # appends the random numbers to n=100 array of ones
    append_random_num = np.append(np.ones(n), random_num)

    # reshapes the data to be transposed
    random_data = np.reshape(append_random_num, (3, n))

    # reshapes the data to be plotted and calulcates dot product
    Xtest = np.transpose(random_data)
    Yhattest = np.matmul(Xtest, w)

    if show_scatter_3 == True:
        # this plot shows the data points which are at x = 1 and y = 1 for the AND function are white.
        # this means that the points are going to pass the AND gate and the black points won't pass the AND gate
        plt.scatter(Xtest[:,1], Xtest[:,2],c=np.reshape(Yhattest,n), edgecolors='black', cmap='gray');
        plt.xlim([0,1]);
        plt.ylim([0,1]);
        plt.show()

    return


def linear_model_xor(show_scatter_1, show_scatter_2):
    #------------------------------------------------------------------------
    #--------------------- Define XOR gate ---------------------
    #------------------------------------------------------------------------
    # define matrices
    X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    Y = np.array([[0], [1], [1], [0]])

    # caluclate inverse of X and X transposed
    dot_product_X = np.matmul(X.T, X)
    inv = np.linalg.inv(dot_product_X)

    # calculate transformations and weights
    dot_product_X_Y = np.matmul(X.T, Y)
    w = np.matmul(inv,  dot_product_X_Y)
    Yhat = np.matmul(X, w)  

    if show_scatter_1 == True:
        plt.scatter(X[:,1], X[:,2],c=np.reshape(Yhat,4), edgecolors='black', cmap='gray');
        plt.show()

    #------------------------------------------------------------------------
    #--------------------- Model Training ---------------------
    #------------------------------------------------------------------------

    # if we have new prediction points, we can use a trained model to predict the Y values
    n = 100
    
    # creates 200 random numbers between 0 and 1 with 100 ones in front
    random_num = np.random.uniform(0, 1, n*2)
    append_random_num = np.append(np.ones(n), random_num)

    # reshapes the data to be plotted and calulcates dot product
    random_data = np.reshape(append_random_num, (3, n))
    Xtest = np.transpose(random_data)
    Yhattest = np.matmul(Xtest, w)

    if show_scatter_2 == True:
        # from the plot, this linear model has failed to represent the XOR gate. This is because
        # the points should be white at (x=0, y=1) and (x=1, y=0) and grey/black everywhere else

        # but the y array we put in gives out 1/2 for all the predictions
        plt.scatter(Xtest[:,1], Xtest[:,2],c=np.reshape(Yhattest,n), edgecolors='black', cmap='gray');
        plt.xlim([0,1]);
        plt.ylim([0,1]);
        plt.savefig(r"Notes 1 Test Figures/1.1 linear model.jpeg")
        plt.show()

    return

#linear_model_and(show_scatter_1=False, show_scatter_2=False, show_scatter_3=True)

linear_model_xor(show_scatter_1=True, show_scatter_2=True)
