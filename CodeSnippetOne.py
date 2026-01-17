
import numpy as np
import argparse
import time

np.random.seed(0)

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
import numpy as np
import argparse
import time

# Keep original small 2-layer network example here; helper utilities
# (neuron, classifier, data generator, fitter) are moved to separate
# modules to keep this file focused on the original task.
from neuron import a
from data import generate_xor_dataset

np.random.seed(0)

# Default toy dataset (will be overridden by synthetic generator in main)
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
    """Mean squared error loss.

    Args:
        params: Parameter vector passed to `f`.
        X: Input features array.
        y: Target array.

    Returns:
        Scalar mean squared error between predictions and targets.
    """
    pred = f(params, X)
    return np.mean((pred - y)**2)


def grad(params, X, y):
    """Numerical gradient (central difference) of `loss` w.r.t. `params`.

    Note: This is an expensive finite-difference approximation. For
    performance and accuracy, consider implementing an analytical
    backpropagation instead of using this function inside large-scale
    optimization loops.
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


def _format_array(a):
    """Format a numpy array or value for human-readable printing."""
    try:
        return np.array2string(np.asarray(a), precision=4, separator=', ')
    except Exception:
        return str(a)


def main():
    """Command-line entry point for training and demos.

    The function parses command-line arguments to control dataset generation,
    optimizer options, and optional demo behaviors (printing data or running
    single-neuron fits). It then either prints the synthetic dataset or runs
    training and displays summary information.
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

    description = (
        "This script trains a tiny 2-layer neural network on a synthetic XOR-like dataset. "
        "By default it generates a dataset of binary inputs (two features) plus a bias column so input size matches L1=3. "
        "It uses a sigmoid activation, mean-squared error loss, and the L-BFGS-B optimizer from SciPy. "
        "The network architecture is: input size L1=3, hidden layer L2=4, output size L3=1.\n\n"
        "When run normally the script prints initial/final loss, training time, and predictions for the test inputs. "
        "Use `--n-samples` and `--noise` to control the synthetic dataset, and `--verbose` / `--maxiter` for optimizer behavior. Use `--show-data` to print the training samples and predictions."
    )
    print(description)
    print('=' * 80)
    print()

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

    print('Starting training of 2-layer neural network on toy dataset')
    print(f'Network sizes: L1={L1}, L2={L2}, L3={L3}')
    print(f'Generating synthetic dataset: n_samples={args.n_samples}, noise={args.noise}, seed={args.seed}')

    # Generate synthetic XOR-like dataset and use it for training
    X, y = generate_xor_dataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)

    init_loss = loss(params, X, y)
    print(f'Initial loss: {init_loss:.6f}')

    t0 = time.time()
    try:
        from scipy import optimize
    except Exception as e:
        raise RuntimeError('scipy is required to run training; install scipy or run only dataset generation functions') from e

    result = optimize.minimize(loss, params, args=(X, y), method='L-BFGS-B', options={'maxiter': args.maxiter, 'disp': args.verbose})
    t1 = time.time()

    if not result.success:
        print('Warning: optimizer did not converge: ', result.message)
    else:
        print('Optimization completed successfully.')

    opt_params = result.x
    final_loss = loss(opt_params, X, y)
    print(f'Final loss: {final_loss:.6f}')
    print(f'Training time: {t1 - t0:.3f}s')

    test = np.array([[0,0,1]])
    pred = f(opt_params, test)
    print('\nTest input:', _format_array(test))
    print('Prediction:', _format_array(pred))

    if args.show_data:
        print('\nAll inputs and predictions:')
        preds = f(opt_params, X)
        for xi, yi, pi in zip(X, y, preds):
            print(f'  X={_format_array(xi)}  target={_format_array(yi)}  pred={_format_array(pi)}')
    else:
        print('\nRun with --show-data to view individual training samples and predictions.')


if __name__ == '__main__':
    main()
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

def _format_array(a):
    """Format a numpy array or value for human-readable printing.

    Args:
        a: Numpy array-like or scalar to format.

    Returns:
        A string representation suitable for concise display (rounded to 4
        decimal places for numeric arrays).
    """
    try:
        return np.array2string(np.asarray(a), precision=4, separator=', ')
    except Exception:
        return str(a)


def main():
    """Command-line entry point for training and demos.

    The function parses command-line arguments to control dataset generation,
    optimizer options, and optional demo behaviors (printing data or running
    single-neuron fits). It then either prints the synthetic dataset or runs
    training and displays summary information.
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

    description = (
        "This script trains a tiny 2-layer neural network on a synthetic XOR-like dataset. "
        "By default it generates a dataset of binary inputs (two features) plus a bias column so input size matches L1=3. "
        "It uses a sigmoid activation, mean-squared error loss, and the L-BFGS-B optimizer from SciPy. "
        "The network architecture is: input size L1=3, hidden layer L2=4, output size L3=1.\n\n"
        "When run normally the script prints initial/final loss, training time, and predictions for the test inputs. "
        "Use `--n-samples` and `--noise` to control the synthetic dataset, and `--verbose` / `--maxiter` for optimizer behavior. Use `--show-data` to print the training samples and predictions."
    )
    print(description)
    print('=' * 80)
    print()

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

    print('Starting training of 2-layer neural network on toy dataset')
    print(f'Network sizes: L1={L1}, L2={L2}, L3={L3}')
    print(f'Generating synthetic dataset: n_samples={args.n_samples}, noise={args.noise}, seed={args.seed}')

    # Generate synthetic XOR-like dataset and use it for training
    X, y = generate_xor_dataset(n_samples=args.n_samples, noise=args.noise, seed=args.seed)

    init_loss = loss(params, X, y)
    print(f'Initial loss: {init_loss:.6f}')

    t0 = time.time()
    try:
        from scipy import optimize
    except Exception as e:
        raise RuntimeError('scipy is required to run training; install scipy or run only dataset generation functions') from e

    result = optimize.minimize(loss, params, args=(X, y), method='L-BFGS-B', options={'maxiter': args.maxiter, 'disp': args.verbose})
    t1 = time.time()

    if not result.success:
        print('Warning: optimizer did not converge: ', result.message)
    else:
        print('Optimization completed successfully.')

    opt_params = result.x
    final_loss = loss(opt_params, X, y)
    print(f'Final loss: {final_loss:.6f}')
    print(f'Training time: {t1 - t0:.3f}s')

    test = np.array([[0,0,1]])
    pred = f(opt_params, test)
    print('\nTest input:', _format_array(test))
    print('Prediction:', _format_array(pred))

    if args.show_data:
        print('\nAll inputs and predictions:')
        preds = f(opt_params, X)
        for xi, yi, pi in zip(X, y, preds):
            print(f'  X={_format_array(xi)}  target={_format_array(yi)}  pred={_format_array(pi)}')
    else:
        print('\nRun with --show-data to view individual training samples and predictions.')


if __name__ == '__main__':
    main()