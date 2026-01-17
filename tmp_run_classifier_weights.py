"""Run fitter, extract weights, and show classifier outputs/accuracy.

Usage: python tmp_run_classifier_weights.py
"""

import numpy as np
from data import generate_xor_dataset
from fitter import fit_single_neuron
from classifier import binary_classifier

np.set_printoptions(precision=6, suppress=True)

X, y = generate_xor_dataset(n_samples=200, noise=0.05, seed=1)
print('Dataset:', X.shape, y.shape)

for act in ('sigmoid', 'relu'):
    print('\n=== Activation:', act, '===')
    params, res = fit_single_neuron(X, y, activation=act, maxiter=500)
    w = params[:-1]
    b = params[-1]
    print('learned params:', params.tolist())

    probs, preds = binary_classifier((w, b), X, activation=act)
    probs = np.asarray(probs).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    acc = (preds == y.reshape(-1)).mean()

    print('  sample probs[:10]:', np.round(probs[:10], 6).tolist())
    print('  sample preds[:10]:', preds[:10].tolist())
    print('  accuracy:', round(float(acc), 4))

print('\nDone.')
