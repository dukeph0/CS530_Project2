"""Fit a single neuron with different activations and print summaries.

Usage: python tmp_fit_activations.py
"""

import numpy as np
from data import generate_xor_dataset
from fitter import fit_single_neuron
from classifier import binary_classifier

np.set_printoptions(precision=6, suppress=True)

X, y = generate_xor_dataset(n_samples=200, noise=0.05, seed=1)
print('Dataset:', X.shape, y.shape)

results = {}
for act in ('sigmoid', 'relu'):
    print('\nFitting activation:', act)
    params, res = fit_single_neuron(X, y, activation=act, maxiter=500)
    print('  success:', getattr(res, 'success', None))
    print('  fun:', getattr(res, 'fun', None))
    print('  params:', params.tolist())

    # params format: [w..., b]
    w = params[:-1]
    b = params[-1]
    probs, preds = binary_classifier((w, b), X, activation=act)
    acc = (preds.reshape(-1) == y.reshape(-1)).mean()
    print('  training accuracy:', round(float(acc), 4))
    results[act] = (params, res, acc)

print('\nAll fits complete.')
