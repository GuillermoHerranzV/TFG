"""Carga y preparacion previa del dataset"""

import numpy as np
import tensorflow as tf


def load_mnist_binary():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = (train_images / 255.0).astype(np.float32)
    test_images = (test_images / 255.0).astype(np.float32)

    train_labels = np.where(train_labels < 5, -1, 1)
    test_labels = np.where(test_labels < 5, -1, 1)
    return (train_images, train_labels), (test_images, test_labels)


def make_subsets(train_images, train_labels, test_images, test_labels, n_train, n_test, seed):
    rng = np.random.default_rng(seed)

    X_train = train_images.reshape(len(train_images), -1)
    X_test = test_images.reshape(len(test_images), -1)

    idx_train = rng.choice(len(X_train), size=n_train, replace=False)
    idx_test = rng.choice(len(X_test), size=n_test, replace=False)

    return {
        "X_train_small": X_train[idx_train],
        "y_train_small": train_labels[idx_train],
        "train_images_small": train_images[idx_train],
        "X_test_small": X_test[idx_test],
        "y_test_small": test_labels[idx_test],
        "test_images_small": test_images[idx_test],
    }
