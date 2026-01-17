import numpy as np
from neuron import neuron


def fit_single_neuron(X, y, activation='sigmoid', method='L-BFGS-B', maxiter=500):
    """Fit a single-neuron classifier to the provided dataset.

    This optimizes a weight vector and bias to minimize an appropriate loss
    for the chosen activation. For `sigmoid` activation it uses binary
    cross-entropy; for other activations it uses mean-squared error.

    Args:
        X: Input array shape (n_samples, n_features).
        y: Target array shape (n_samples, 1) or (n_samples,).
        activation: 'sigmoid' or 'relu' to select activation.
        method: Optimizer method passed to `scipy.optimize.minimize`.
        maxiter: Maximum iterations for the optimizer.

    Returns:
        params: 1D numpy array of length n_features+1 containing [w..., b].
        result: `OptimizeResult` returned by `scipy.optimize.minimize`.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n_features = X.shape[1]

    def _loss(params):
        w = params[:-1]
        b = params[-1]
        probs = neuron(X, w, b, activation=activation)
        probs = np.asarray(probs).reshape(-1)
        if activation == 'sigmoid':
            eps = 1e-9
            probs = np.clip(probs, eps, 1 - eps)
            return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        else:
            return np.mean((probs - y) ** 2)

    init = np.zeros(n_features + 1)
    # Import scipy.optimize here to avoid requiring SciPy at module import time
    try:
        from scipy import optimize
    except Exception as e:
        raise RuntimeError('scipy is required to fit a neuron; install scipy or avoid calling fit_single_neuron') from e

    result = optimize.minimize(_loss, init, method=method, options={'maxiter': maxiter})
    return result.x, result
