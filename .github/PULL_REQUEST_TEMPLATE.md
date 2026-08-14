# Summary

<!-- What changed and why. Link the issue it closes, if any. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New model family
- [ ] New study, ablation or metric
- [ ] Documentation
- [ ] Tooling, CI or packaging
- [ ] Refactor with no behavioural change

## Effect on reported results

<!--
Does this move any reported cost, recall, precision or calibration number?
If yes, say which experiment produced the new figure and paste the before/after.
If no, say "no change to reported results" explicitly — reviewers rely on this.
-->

## Checks

- [ ] `make lint` passes (ruff + mypy --strict)
- [ ] `make test` passes
- [ ] New or changed behaviour is covered by a test
- [ ] Any new optional dependency is in its own Poetry group and imported lazily
- [ ] Notebook outputs are cleared

## Research protocol

- [ ] The official UCI test set was not used for any modelling decision
- [ ] The fit / tune / calibration / threshold roles remain separate
- [ ] Resampling, if any, stays inside a fit-only pipeline
- [ ] Any new model reports `J = 10*FP + 500*FN`
- [ ] Thresholds are learned on the threshold subset, not assumed

## Notes for the reviewer

<!-- Anything you are unsure about, or that deserves a closer look. -->
