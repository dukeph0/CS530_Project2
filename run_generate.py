"""Utility: generate XOR-like dataset and offer a small CLI demo.

This module provides a dataset generator used by the demo scripts
and includes a tiny `__main__` example to print a few samples.
"""

import numpy as np

def generate_xor_dataset(n_samples=100, noise=0.0, seed=None):
    """Generate a synthetic XOR-like dataset.

    Produces inputs with two binary features plus a bias column so the
    input dimensionality is 3. This is a standalone copy used for quick
    runs; the same function also exists in `CodeSnippetOne.py`.

    Args:
        n_samples: Number of samples to generate.
        noise: Probability of flipping each label (0.0..1.0).
        seed: Optional random seed for reproducibility.

    Returns:
        X: array shape (n_samples, 3) where columns are [x1, x2, 1].
        y: array shape (n_samples, 1) binary XOR labels.
    """
    rng = np.random.RandomState(seed)
    x1 = rng.randint(0, 2, size=n_samples)
    x2 = rng.randint(0, 2, size=n_samples)
    labels = (x1 ^ x2).astype(int)
    if noise > 0:
        flip = rng.rand(n_samples) < float(noise)
        labels = labels ^ flip.astype(int)
    X = np.stack([x1, x2, np.ones(n_samples)], axis=1)
    y = labels.reshape(-1, 1)
    return X, y

if __name__ == '__main__':
    X, y = generate_xor_dataset(n_samples=10, noise=0.1, seed=42)
    print('X.shape =', X.shape)
    print('y.shape =', y.shape)
    print('\nX[:10]:')
    print(X[:10])
    print('\ny[:10]:')
    print(y[:10].reshape(-1))
