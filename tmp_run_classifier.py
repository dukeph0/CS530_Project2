"""Demo: fit a single neuron and print fit + classifier summaries.

Usage: python tmp_run_classifier.py
"""

from data import generate_xor_dataset
from fitter import fit_single_neuron
from classifier import binary_classifier
import numpy as np

# Generate dataset matching previous run
X, y = generate_xor_dataset(n_samples=50, noise=0.05, seed=1)

def run_and_print(act):
    """Fit a single neuron using `act` activation and print summary.

    Args:
        act: Activation function name to pass to `fit_single_neuron` and
             `binary_classifier` (e.g., 'sigmoid' or 'relu').

    Side effects:
        Prints fit status, final loss, training accuracy, and example
        probabilities/predictions to stdout.
    """
    print(f"\n--- Activation: {act} ---")
    params, res = fit_single_neuron(X, y, activation=act, maxiter=200)
    print('Fit success:', getattr(res, 'success', None))
    print('Final loss:', getattr(res, 'fun', None))
    probs, preds = binary_classifier(params, X, activation=act)
    acc = np.mean(preds.reshape(-1) == y.reshape(-1))
    print('Accuracy on training set:', acc)
    print('Sample probs[:10]:', np.round(probs[:10].reshape(-1), 4))
    print('Sample preds[:10] :', preds[:10].reshape(-1))

for activation in ('sigmoid', 'relu'):
    try:
        run_and_print(activation)
    except Exception as e:
        print('Error running for', activation, e)
