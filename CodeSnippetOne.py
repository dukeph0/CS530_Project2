
import numpy as np
from scipy import optimize
from scipy.special import expit

np.random.seed(0)

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0], [1], [1], [0]])

L1 = 3
L2 = 4
L3 = 1

w1 = np.random.randn(L1 * L2)
w2 = np.random.randn(L2 * L3)
b1 = np.zeros(L2)
b2 = np.zeros(L3)

params = np.concatenate([w1, b1, w2, b2])

def a(x):
    """Sigmoid activation function.

    Args:
        x: Numeric input (numpy array or scalar) representing logits.

    Returns:
        Elementwise sigmoid of `x` with the same shape as `x`.
    """
    return expit(x)

def f(params, X):
    w1 = params[:L1*L2].reshape(L1, L2)
    b1 = params[L1*L2:L1*L2+L2]
    w2 = params[L1*L2+L2:L1*L2+L2+L2*L3].reshape(L2, L3)
    b2 = params[-L3:]
    
    """Forward pass of a small 2-layer neural network.

    Parameters are packed into a 1D vector `params` in the order:
    w1 (L1*L2), b1 (L2), w2 (L2*L3), b2 (L3).

    Args:
        params: 1D numpy array containing flattened weights and biases.
        X: Input data array of shape (n_samples, L1).

    Returns:
        Network outputs after sigmoid activation with shape (n_samples, L3).
    """

    h = a(X.dot(w1) + b1)
    out = a(h.dot(w2) + b2)
    return out

def loss(params, X, y):
    pred = f(params, X)
    """Mean squared error loss.

    Args:
        params: Parameter vector passed to `f`.
        X: Input features array.
        y: Target array.

    Returns:
        Scalar mean squared error between predictions and targets.
    """

    return np.mean((pred - y)**2)

def grad(params, X, y):
    """Numerical gradient (central difference) of `loss` w.r.t. `params`.

    Note: This is an expensive finite-difference approximation. For
    performance and accuracy, consider implementing an analytical
    backpropagation instead of using this function inside large-scale
    optimization loops.

    Args:
        params: 1D numpy array of parameters.
        X: Input features array.
        y: Target array.

    Returns:
        1D numpy array with the same shape as `params` containing
        the approximate gradient of `loss`.
    """

    eps = 1e-7
    g = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        g[i] = (loss(params_plus, X, y) - loss(params_minus, X, y)) / (2*eps)
    return g

result = optimize.minimize(loss, params, args=(X, y), method='L-BFGS-B', options={'maxiter': 1000, 'disp': True})

opt_params = result.x

test = np.array([[0,0,1]])
pred = f(opt_params, test)
print("Test:", pred)

print("\nAll:")
print(f(opt_params, X))