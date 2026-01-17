import numpy as np


def a(x):
    """Sigmoid activation function.

    Args:
        x: Numeric input (numpy array or scalar) representing logits.

    Returns:
        Elementwise sigmoid of `x` with the same shape as `x`.
    """
    return 1.0 / (1.0 + np.exp(-x))


def neuron(x, w, b, activation='sigmoid'):
    """Compute a single neuron output with selectable activation.

    Args:
        x: Input array of shape (n_samples, n_features) or (n_features,).
        w: Weight vector of shape (n_features,).
        b: Bias scalar.
        activation: 'sigmoid' or 'relu' to select activation function.

    Returns:
        Numpy array of neuron outputs.
    """

    z = np.dot(x, w) + b
    if activation == 'sigmoid':
        return a(z)
    elif activation == 'relu':
        return np.maximum(0, z)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
