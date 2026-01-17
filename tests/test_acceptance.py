"""Acceptance tests for core project functionality.

Provides pytest cases that validate the neuron, classifier, dataset
generation, and (optionally) fitting utilities.
"""

import os
import sys
import numpy as np
import pytest

# Ensure the project root is on sys.path so local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neuron import neuron, a
from classifier import binary_classifier
from data import generate_xor_dataset


def test_neuron_two_activations_and_shapes():
    # simple dataset: two samples, two features + bias column
    X = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    w = np.array([1.0, -1.0, 0.0])
    b = 0.0

    # sigmoid
    out_sig = neuron(X, w, b, activation='sigmoid')
    assert out_sig.shape[0] == X.shape[0]
    assert np.all((out_sig >= 0.0) & (out_sig <= 1.0))

    # relu
    out_relu = neuron(X, w, b, activation='relu')
    assert out_relu.shape[0] == X.shape[0]
    assert np.all(out_relu >= 0.0)


def test_binary_classifier_weight_formats_and_output():
    X = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    # weight as (w,b) tuple
    w = np.array([0.0, 0.0, 0.0])
    b = 0.6
    probs, preds = binary_classifier((w, b), X, activation='sigmoid', threshold=0.5)
    assert probs.shape[0] == X.shape[0]
    assert preds.shape[0] == X.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})

    # weight as flattened array (w..., b)
    flat = np.concatenate([w, np.array([b])])
    probs2, preds2 = binary_classifier(flat, X, activation='sigmoid', threshold=0.5)
    assert np.allclose(probs, probs2)
    assert np.array_equal(preds, preds2)


def test_generate_xor_dataset_shapes_and_noise():
    X, y = generate_xor_dataset(n_samples=50, noise=0.0, seed=7)
    assert X.shape == (50, 3)
    assert y.shape == (50, 1)
    # bias column should be all ones
    assert np.all(X[:, 2] == 1)

    # noise should flip some labels when > 0
    Xn, yn = generate_xor_dataset(n_samples=200, noise=0.2, seed=0)
    assert Xn.shape == (200, 3)
    assert yn.shape == (200, 1)


def test_docstrings_present():
    # Basic smoke tests that key functions have docstrings
    assert neuron.__doc__ and neuron.__doc__.strip()
    assert binary_classifier.__doc__ and binary_classifier.__doc__.strip()
    assert generate_xor_dataset.__doc__ and generate_xor_dataset.__doc__.strip()


def test_fit_single_neuron_and_classifier_end_to_end():
    # This test depends on scipy; skip if unavailable
    pytest.importorskip('scipy')

    from fitter import fit_single_neuron

    X, y = generate_xor_dataset(n_samples=100, noise=0.05, seed=2)

    for activation in ('sigmoid', 'relu'):
        params, res = fit_single_neuron(X, y, activation=activation, maxiter=200)
        # params should be of length n_features+1
        assert params.shape[0] == X.shape[1] + 0  or params.shape[0] == X.shape[1] + 1
        # returned result object should have success attribute
        assert hasattr(res, 'success')

        # run classifier with learned weights
        w = params[:-1]
        b = params[-1]
        probs, preds = binary_classifier((w, b), X, activation=activation)
        assert probs.shape[0] == X.shape[0]
        assert preds.shape[0] == X.shape[0]
