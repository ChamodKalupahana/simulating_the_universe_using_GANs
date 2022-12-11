import time

import numpy as np
print("numpy: {}".format(np.__version__))

import matplotlib.pyplot as plt
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

# test time taken to install
start_time = time.time()
import tensorflow as tf
print("tensorflow: {}".format(tf.version.VERSION))

end_time = time.time()
time_taken = end_time - start_time
print('Time taken =', str(time_taken) + 'secs')

# at first, tensorflow took around 15 seconds to import
# for repeating tests, tensorflow took around 5 seconds to import
# assume, tensorflow modules entered cache?

print('tensorflow install correctly')