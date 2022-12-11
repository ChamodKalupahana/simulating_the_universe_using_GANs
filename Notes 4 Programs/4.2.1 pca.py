import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf # around 5.6 seconds to import tensorflow

import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


(X_train, Y_train), (X_test, Y_test) =  tf.keras.datasets.mnist.load_data()

# normalise data
X_train = X_train/ 255
X_test = X_test/ 255

# reshape data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2])).astype(np.float32)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2])).astype(np.float32)

# get pca components
pca = PCA(n_components=3)
pca.fit(X_train)

X_encoded = pca.transform(X_test)

plt.figure(figsize=(8,6))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=Y_test, edgecolor='none', alpha=1.0, 
            cmap=plt.get_cmap('jet', 10), s=6)
plt.colorbar()
plt.xlabel('PCA component 1', fontsize=14)
plt.ylabel('PCA component 2', fontsize=14)
plt.show()

print('Hello World')