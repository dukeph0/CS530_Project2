"""Quick demo: generate a small XOR dataset and print shapes/samples.

Usage: python tmp_run_snip.py
"""

from CodeSnippetOne import generate_xor_dataset
X, y = generate_xor_dataset(n_samples=10, noise=0.1, seed=42)
print('X.shape =', X.shape)
print('y.shape =', y.shape)
print('\nX[:10]:')
print(X[:10])
print('\ny[:10]:')
print(y[:10].reshape(-1))
