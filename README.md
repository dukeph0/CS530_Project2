# Tiny Neural Demo + Utilities

This repository contains a small Python project used for teaching and quick experimentation with single-neuron models and a tiny 2-layer demo network. It was developed as a compact demonstration for local development and a GitHub project page.

Contents
- `CodeSnippetOne.py` – A small 2-layer network demo (forward, MSE loss, numerical grad) plus a CLI to generate a toy XOR-like dataset and run a short training demo (requires SciPy for optimization).
- `neuron.py` – Single-neuron primitives and stable sigmoid activation.
- `classifier.py` – `binary_classifier` wrapper that accepts multiple weight formats and yields probabilities + binary predictions.
- `data.py` / `run_generate.py` – Synthetic XOR-like dataset generator used by demos and tests.
- `fitter.py` – Small helper to fit a single-neuron classifier using SciPy (lazily imported so the rest of the repo works without SciPy).
- `CodeSnippetTwo.py` – Small payroll example utility used as an additional demo.
- `tmp_*.py` – Temporary helper/demo scripts used during development (convenient to run locally).
- `tests/test_acceptance.py` – Pytest acceptance tests for core functionality.
- `requirements.txt` – Minimal dependencies used for development/testing.

Quick start

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (SciPy is optional but required to run training demos):

```bash
pip install -r requirements.txt
```

3. Run quick integration smoke tests/demos:

```bash
# short smoke test that exercises many modules
python tmp_run_all_tests.py

# small dataset printout
python tmp_run_generate.py

# run the small 2-layer training demo (requires SciPy)
python CodeSnippetOne.py
```

4. Run the project tests:

```bash
pytest -q
```

Notes
- Training routines (`fitter.py` and the optimizer call inside `CodeSnippetOne.py`) require SciPy. The dataset generation, neuron, classifier, and most demos work with only NumPy.
- The `tmp_*.py` scripts are small convenience demos used during development; feel free to delete or move them into a `demos/` folder if you prefer a tidier repository layout.

Contributing and next steps
- If you want a GitHub Actions workflow to run `pytest` on push/PR, add `.github/workflows/python-tests.yml` (one has been included in this change).
- Consider moving the `tmp_*.py` scripts into a `demos/` directory for clarity.

License
- No license file is included by default. If you want a license, tell me which one and I will add it.