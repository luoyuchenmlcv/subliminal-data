# Subliminal Data

Code for reproducing the steering-vector and Fisher-spectrum figures in this
project. Each figure has one executable entry point under
`figure_reproduction/`.

## Setup

Python 3.11 is recommended.

```bash
uv sync --frozen
```

Model locations and experiment constants are configured in
`figure_reproduction/paper.json`. Update the local model paths before running
on a different machine.

## Running figures

Inspect the generated commands without launching an experiment:

```bash
uv run --frozen --no-dev python figure_reproduction/figure_01.py --dry-run
```

Run an individual stage:

```bash
uv run --frozen --no-dev python figure_reproduction/figure_03.py \
  --stage prepare --models qwen --traits cat
```

Available stages are `prepare`, `run`, `aggregate`, `plot`, and `all`. Outputs
are written to `reproduction_outputs/` by default.

See [`figure_reproduction/README.md`](figure_reproduction/README.md) for the
figure-level organization and exact experiment conventions.

## Tests

```bash
uv run --frozen --no-dev pytest -q test/test_figure_reproduction.py
PYTHONPATH=. uv run --frozen --no-dev pytest -q \
  sv_scripts/tests/test_steering_vector_pipeline.py
```

## Acknowledgments

The carrier-data utilities build on the open-source
[`lmb-freiburg/divergence-tokens`](https://github.com/lmb-freiburg/divergence-tokens)
codebase. Its license and notices are retained in this repository.
