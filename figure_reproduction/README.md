# Figure-level reproduction

Each paper figure has one executable entry point.  The entry points orchestrate
the existing steering-vector implementations

```bash
python figure_reproduction/figure_01.py --dry-run
python figure_reproduction/figure_02.py --stage prepare --models qwen
python figure_reproduction/figure_03.py --stage all --models qwen
python figure_reproduction/figure_04.py --stage run --models qwen
python figure_reproduction/figure_05.py --stage plot
```

All generated state is placed below `--output-root` (default:
`reproduction_outputs`). Existing exploratory artifacts under `sv_scripts/data`
and `sv_single_scripts/data` are never overwritten.

Stages are `prepare`, `run`, `aggregate`, `plot`, and `all`.  Every command also
supports `--dry-run`, resumable terminal-artifact checks, model/trait subsets,
and disabled/offline/online W&B modes.  Exact paper constants live in
`paper.json`; in particular teacher extraction uses 200 updates, Figure 3 uses
500 carriers and exactly 10,000 optimizer steps, and the Fisher uses 30,000
clean contexts with four resamples per context.

Figure 1 preserves the paper runners' optimization horizons: hard-label
iterative recovery uses 1,000 updates for Qwen and 500 for Gemma, while the
soft-label iterative recovery uses 10,000 updates for both models. Single-step
conditions use all 30,000 carrier sequences once.

The Fisher uses the sequence-equal Rademacher estimator stated in Appendix
A.4. Carrier completion lengths are nevertheless recorded so token-count and
sequence-count interpretations remain auditable.
