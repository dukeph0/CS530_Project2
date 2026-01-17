"""Quick integration smoke tests for core modules.

Runs a short sequence exercising dataset generation, neuron,
classifier, fitter (if available), and payroll utilities.

Usage: python tmp_run_all_tests.py
"""

import numpy as np

print('Running quick integration tests...')

# Test data generator
from data import generate_xor_dataset
X, y = generate_xor_dataset(n_samples=6, noise=0.0, seed=123)
print('generate_xor_dataset -> X.shape, y.shape:', X.shape, y.shape)
print('X[:3]:', X[:3].tolist())
print('y[:3]:', y[:3].reshape(-1).tolist())

# Test neuron
from neuron import neuron
w = np.array([1.0, -1.0, 0.0])
b = 0.0
print('\nneuron test (sigmoid) on first 3 samples:')
print(neuron(X[:3], w, b, activation='sigmoid'))
print('neuron test (relu):')
print(neuron(X[:3], w, b, activation='relu'))

# Test classifier
from classifier import binary_classifier
w_all = np.array([1.0, -1.0, 0.0])  # X has 3 columns (x1,x2,1) -> w length 3, bias passed as tuple
probs, preds = binary_classifier((w_all, 0.0), X, activation='sigmoid')
print('\nbinary_classifier probs[:5]:', np.round(probs[:5].reshape(-1),4).tolist())
print('binary_classifier preds[:5]:', preds[:5].reshape(-1).tolist())

# Test CodeSnippetOne forward/loss/grad
import CodeSnippetOne as cs1
params = cs1.params
smallX, smally = generate_xor_dataset(n_samples=4, noise=0.0, seed=0)
print('\nCodeSnippetOne f output shape:', cs1.f(params, smallX).shape)
print('loss(params, smallX, smally):', cs1.loss(params, smallX, smally))
print('grad(params, smallX, smally) length:', cs1.grad(params, smallX, smally).shape)

# Test fitter if available
try:
    from fitter import fit_single_neuron
    print('\nRunning fit_single_neuron (may require scipy)')
    p, res = fit_single_neuron(smallX, smally, activation='sigmoid', maxiter=50)
    print('fit success:', getattr(res, 'success', None), 'fun:', getattr(res, 'fun', None))
except Exception as e:
    print('fit_single_neuron skipped / failed:', e)

# Test payroll
from CodeSnippetTwo import process_employee_data
print('\nPayroll test:')
print(process_employee_data('2001', 45, 20.0, 'IT'))

print('\nAll quick tests complete.')
