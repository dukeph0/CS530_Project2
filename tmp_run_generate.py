"""Small script to print a few rows of the XOR dataset.

Usage: python tmp_run_generate.py
"""

from data import generate_xor_dataset

print('Generating 10 samples (noise=0.1, seed=42)')
X, y = generate_xor_dataset(n_samples=10, noise=0.1, seed=42)
print('X.shape =', X.shape)
print('y.shape =', y.shape)
print('\nFirst 10 X rows:')
for row in X[:10]:
    print(row.tolist())
print('\nFirst 10 y labels:')
print(y[:10].reshape(-1).tolist())
