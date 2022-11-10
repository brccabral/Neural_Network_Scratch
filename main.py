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
