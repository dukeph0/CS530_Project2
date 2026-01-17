
"""Small 2-layer network demo (clean single-copy implementation).

This module provides a tiny 2-layer network, loss, numerical grad, and a
CLI to generate data or run a small training demo using SciPy's L-BFGS-B.
"""

import numpy as np
import argparse
import time

from neuron import a
from data import generate_xor_dataset

np.random.seed(0)

# Default toy dataset (will be overridden by synthetic generator in main)
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])

L1 = 3
L2 = 4
L3 = 1

w1 = np.random.randn(L1 * L2)
w2 = np.random.randn(L2 * L3)
b1 = np.zeros(L2)
b2 = np.zeros(L3)

params = np.concatenate([w1, b1, w2, b2])


def f(params, X):
    """Forward pass for the 2-layer network.

    Parameters are packed into `params` in the order: w1, b1, w2, b2.

    Args:
        params: 1D numpy array of network parameters.
        X: Input array of shape (n_samples, L1).

    Returns:
        Network outputs with shape (n_samples, L3).
    """

    w1 = params[: L1 * L2].reshape(L1, L2)
    b1 = params[L1 * L2 : L1 * L2 + L2]
    w2 = params[L1 * L2 + L2 : L1 * L2 + L2 + L2 * L3].reshape(L2, L3)
    b2 = params[-L3:]

    h = a(X.dot(w1) + b1)
    out = a(h.dot(w2) + b2)
    return out


def loss(params, X, y):
    """Mean squared error between network predictions and targets.

    Args:
        params: Parameter vector passed to `f`.
        X: Input features array.
        y: Target array.

    Returns:
        Scalar mean squared error.
    """

    pred = f(params, X)
    return np.mean((pred - y) ** 2)


def grad(params, X, y):
    """Numerical gradient (central difference) of `loss` w.r.t. `params`.

    Returns:
        1D numpy array with the same shape as `params` containing
        the approximate gradient.
    """

    eps = 1e-7
    g = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        g[i] = (loss(params_plus, X, y) - loss(params_minus, X, y)) / (2 * eps)
    return g


def _format_array(a):
    """Return a concise string representation for arrays/scalars.

    Rounds numeric arrays to 4 decimal places for compact display.
    """

    try:
        return np.array2string(np.asarray(a), precision=4, separator=', ')
    except Exception:
        return str(a)


def main():
    """Command-line entrypoint: generate data or train the 2-layer network.

    Parses CLI args (dataset size, noise, optimizer options) and either
    prints a generated dataset or runs the L-BFGS-B optimizer to train
    the small network, printing summary information.
    """

    parser = argparse.ArgumentParser(description='Train small 2-layer NN on toy dataset with clear UI messages.')
    parser.add_argument('--maxiter', type=int, default=1000, help='Max iterations for optimizer')
    parser.add_argument('--verbose', action='store_true', help='Show optimizer output')
    parser.add_argument('--n-samples', type=int, default=100, help='Number of synthetic samples to generate')
    parser.add_argument('--noise', type=float, default=0.0, help='Label flip noise probability')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for synthetic data')
    parser.add_argument('--show-data', action='store_true', help='Show the training dataset and predictions')
    parser.add_argument('--print-data', action='store_true', help='Generate and print a small synthetic dataset then exit')
    args = parser.parse_args()

    if args.print_data:
        Xp, yp = generate_xor_dataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)
        print(f'Generated dataset (n_samples={args.n_samples}, noise={args.noise}, seed={args.seed})')
        print('X.shape =', Xp.shape)
        print('y.shape =', yp.shape)
        print('\nFirst rows of X:')
        print(Xp[:10])
        print('\nFirst rows of y:')
        print(yp[:10].reshape(-1))
        return

    X_train, y_train = generate_xor_dataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)

    init_loss = loss(params, X_train, y_train)
    print(f'Initial loss: {init_loss:.6f}')

    t0 = time.time()
    try:
        from scipy import optimize
    except Exception as e:
        raise RuntimeError('scipy is required to run training; install scipy or run only dataset generation functions') from e

    result = optimize.minimize(loss, params, args=(X_train, y_train), method='L-BFGS-B', options={'maxiter': args.maxiter, 'disp': args.verbose})
    t1 = time.time()

    if not result.success:
        print('Warning: optimizer did not converge: ', result.message)
    else:
        print('Optimization completed successfully.')

    opt_params = result.x
    final_loss = loss(opt_params, X_train, y_train)
    print(f'Final loss: {final_loss:.6f}')
    print(f'Training time: {t1 - t0:.3f}s')

    test = np.array([[0, 0, 1]])
    pred = f(opt_params, test)
    print('\nTest input:', _format_array(test))
    print('Prediction:', _format_array(pred))


if __name__ == '__main__':
    main()