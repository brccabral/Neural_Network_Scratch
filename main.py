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
# data_dev = data[0:1000].T  # transpose to make math easier
# Y_dev = data_dev[0]
# X_dev = data_dev[1:n]
# X_dev = X_dev / 255.0

# %%
# data_train = data[1000:m].T  # transpose to make math easier
# data = data[:15][:]
data_train = data.T  # transpose to make math easier
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0


# %%
def init_params() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate weights and biases"""
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# %%
def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(Z, 0)


# %%
def softmax(Z: np.array) -> np.ndarray:
    return np.exp(Z) / sum(np.exp(Z))


# %%
def forward_prop(
    W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Z1 = W1.dot(X) + b1  # W1 10,784 ||| X 784,60000 ||| W.X 10,60000
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
    one_hot_Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = Y.size
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1: np.ndarray = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
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
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


# %%
def get_predictions(A2: np.ndarray) -> np.ndarray:
    return np.argmax(A2, 0)


# %%
def get_accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


# %%
def gradient_descent(
    X: np.ndarray, Y: np.ndarray, iterations: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    W1, b1, W2, b2 = init_params()
    one_hot_Y = one_hot(Y)
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y, one_hot_Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print(f"Generation: {i}\tAccuracy: {get_accuracy(get_predictions(A2), Y)}")
    print(f"Final\tAccuracy: {get_accuracy(get_predictions(A2), Y)}")
    return W1, b1, W2, b2


# %%
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)


# %%
def make_predictions(
    X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)


# %%%
def test_prediction(
    index: int,
    X: np.ndarray,
    Y: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
):
    current_image = X[:, index, None]
    prediction = make_predictions(X[:, index, None], W1, b1, W2, b2)
    label = Y[index]
    print(f"Prediction: {prediction[0]}\tLabel: {label}")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()


# %%
test = pd.read_csv("./mnist/mnist_test.csv")
test = np.array(test)
data_test = test.T  # transpose to make math easier
Y_test: np.ndarray = data_test[0]
X_test: np.ndarray = data_test[1:n]
X_test: np.ndarray = X_test / 255.0

# %%
test_prediction(0, X_test, Y_test, W1, b1, W2, b2)
test_prediction(1, X_test, Y_test, W1, b1, W2, b2)
test_prediction(2, X_test, Y_test, W1, b1, W2, b2)
test_prediction(3, X_test, Y_test, W1, b1, W2, b2)

# %%
test_prediction(int(np.random.rand() * Y_test.size), X_test, Y_test, W1, b1, W2, b2)

# %%
test_predictions = make_predictions(X_test, W1, b1, W2, b2)
get_accuracy(test_predictions, Y_test)

# %%
