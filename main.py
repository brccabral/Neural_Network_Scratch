# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("./mnist/mnist_train.csv")

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
def init_params() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate weights and biases"""
    W1 = np.random.randn(10, 784) - 0.5
    b1 = np.random.randn(10, 1) - 0.5
    W2 = np.random.randn(10, 10) - 0.5
    b2 = np.random.randn(10, 1) - 0.5
    return W1, b1, W2, b2


# %%
def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)


# %%
def softmax(Z: np.array) -> np.array:
    return np.exp(Z) / np.sum(np.exp(Z))


# %%
def forward_prop(
    W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# %%
def one_hot(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # create matrix with correct size
    # np.arange(Y.size) - will return a list from 0 to Y.size
    # Y - will contain the column index to set the value of 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T  # transpose


# %%
def deriv_ReLU(Z: np.ndarray) -> np.ndarray:
    return Z > 0


# %%
def back_prop(
    Z1: np.ndarray,
    A1: np.ndarray,
    Z2: np.ndarray,
    A2: np.ndarray,
    W2: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 2)
    dZ1: np.ndarray = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 2)
    return dW1, db1, dW2, db2


# %%
def update_params(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    dW1: np.ndarray,
    db1: np.ndarray,
    dW2: np.ndarray,
    db2: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# %%
def get_predictions(A2) -> np.ndarray:
    return np.argmax(A2, 0)


# %%
def get_accuracy(predictions, Y) -> float:
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
