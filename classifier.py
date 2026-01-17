import numpy as np
from neuron import neuron


def binary_classifier(weights, X, activation='sigmoid', threshold=0.5):
    """Simple binary classifier using a single neuron.

    The `weights` parameter may be provided in one of these formats:
      - 1D list/array of length n_features+1 where the last element is the bias
      - tuple/list of two items `(w, b)` where `w` is the weight vector and `b` is the bias scalar

    Args:
        weights: Weight specification as described above.
        X: Input dataset, shape (n_samples, n_features) or (n_features,) for single sample.
        activation: Activation function to use ('sigmoid' or 'relu').
        threshold: Decision threshold applied to the activation output to produce binary labels.

    Returns:
        probs: Activation outputs (probabilities/scores).
        preds: Binary predictions (0 or 1) as a numpy array.
    """

    # Normalize inputs
    X = np.asarray(X)

    # Parse weights
    if isinstance(weights, (list, tuple)) and len(weights) == 2:
        w = np.asarray(weights[0])
        b = float(weights[1])
    else:
        w_arr = np.asarray(weights)
        if w_arr.ndim == 1 and w_arr.size == (X.shape[-1] + 1):
            w = w_arr[:-1]
            b = float(w_arr[-1])
        else:
            raise ValueError('weights must be length n_features+1 or (w, b)')

    probs = neuron(X, w, b, activation=activation)
    preds = (probs >= threshold).astype(int)
    return probs, preds
