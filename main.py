# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv('./mnist/mnist_train.csv')

# %%
data.head()

# %%
data = np.array(data)

# %%
m, n = data.shape
np.random.shuffle(data)  # avoid overfitting

# %%
data_dev = data[0:1000].T  # transpose to make math easier
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

# %%
data_train = data[1000:].T  # transpose to make math easier
Y_train = data_train[0]
X_train = data_train[1:n]

# %%
