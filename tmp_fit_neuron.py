"""Temporary helper: fit a single neuron on a synthetic XOR dataset.

This script calls `generate_xor_dataset` and `fit_single_neuron` from
`CodeSnippetOne.py` and prints the learned parameters and optimizer
summary for two activations ('sigmoid' and 'relu').

Usage:
    python tmp_fit_neuron.py

This is a convenience script used during development and can be removed
or integrated into the main CLI as needed.
"""

from data import generate_xor_dataset
from fitter import fit_single_neuron


X, y = generate_xor_dataset(n_samples=50, noise=0.05, seed=1)

for act in ('sigmoid', 'relu'):
    print(f"\nFitting single neuron with activation='{act}'")
    try:
        params, res = fit_single_neuron(X, y, activation=act, maxiter=200)
    except Exception as e:
        print('Error:', e)
        continue
    print('Success:', getattr(res, 'success', None))
    print('Message:', getattr(res, 'message', None))
    print('Final loss:', getattr(res, 'fun', None))
    print('Learned params (w..., b):', params)
