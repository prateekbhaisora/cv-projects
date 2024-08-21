from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import os

mnist_dataset = fetch_openml('mnist_784')
X = mnist_dataset.data
y = mnist_dataset.target

TRAIN_RATIO = 0.6
TEST_RATIO = 0.4

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, random_state=42)

train_dir = "train data"
test_dir = "test data"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

np.save(os.path.join(train_dir, "X_train.npy"), X_train)
np.save(os.path.join(train_dir, "y_train.npy"), y_train)

np.save(os.path.join(test_dir, "X_test.npy"), X_test)
np.save(os.path.join(test_dir, "y_test.npy"), y_test)

# # This code was just for debugging purposes

# X_train = np.load("train data/X_train.npy", allow_pickle=True)
# y_train = np.load("train data/y_train.npy", allow_pickle=True)

# X_test = np.load("test data/X_test.npy", allow_pickle=True)
# y_test = np.load("test data/y_test.npy", allow_pickle=True)

# import matplotlib.pyplot as plt
# import matplotlib

# x, y = X_train, X_train

# some_digit = x[28]
# some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it

# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
#            interpolation='nearest')
# plt.axis("off")
# plt.show()