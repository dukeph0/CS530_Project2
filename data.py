import numpy as np


def generate_xor_dataset(n_samples=100, noise=0.0, seed=None):
    """Generate a synthetic XOR-like dataset suitable for the small network.

    Produces inputs with two binary features plus a bias column (1) so the
    input dimensionality matches L1==3 used elsewhere in the script.

    Args:
        n_samples: Number of samples to generate.
        noise: Probability in [0,1] of flipping each label (introduce noise).
        seed: Optional random seed for reproducibility.

    Returns:
        X: array shape (n_samples, 3) where columns are [x1, x2, 1].
        y: array shape (n_samples, 1) with binary XOR labels.
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
