# Three-Method Pilot Fairness Audit

## Scope

This document records the pilot-stage fairness conditions for matched search
seeds 2701 and 2702. The three methods share the same real NASNet search space,
training pipeline, and real CNN training budget. Only the intended selection
and repeat-policy components differ.

## Fairness table

| Item | RE | SA-RE | RS-SA-RE |
|---|---|---|---|
| NASNet search space | same | same | same |
| Network depth/base filters (N/F) | 3/24 | 3/24 | 3/24 |
| Training epochs per real evaluation | 5 | 5 | 5 |
| Dataset and split | CIFAR-10, same 45k/5k split | same | same |
| Split seed | 20260823 | 20260823 | 20260823 |
| Trainer | shared trainer | same | same |
| Fitness used by search | first-seed final validation accuracy | same | same |
| Population size (P) | 20 | 20 | 20 |
| Tournament size (S) | 5 | 5 | 5 |
| Tournament sampling | with replacement | same | same |
| Aging | FIFO | FIFO | FIFO |
| Mutation operator/distribution | shared | same | same |
| Candidate count (K) | 1 | 5 | 5 |
| Child selection | direct mutated child | `argmax(mu_hat)` | `argmax(mu_hat - lambda * d_hat)` |
| Pilot lambda | not applicable | not applicable | 1.0 |
| Repeat evaluation | none | none | scheduled |
| Pilot repeat policy | not applicable | not applicable | `warmup_pairs=4`, `repeat_interval=4` |
| Real CNN training budget | B=30 | B=30 | B=30, including repeats |

## Intended method differences

- RE creates and directly evaluates one mutated child per evolutionary step.
- SA-RE creates `K=5` candidates and evaluates the child with maximum predicted
  validation accuracy, `argmax(mu_hat)`.
- RS-SA-RE creates `K=5` candidates and evaluates the child with maximum
  stability-aware score, `argmax(mu_hat - lambda * d_hat)`.
- RS-SA-RE schedules real repeat evaluations to obtain paired instability
  labels. RE and SA-RE do not train repeats.

These differences define the methods and must not be “equalized” in the
fairness audit.

## Real-budget and search-fitness definitions

Every real CNN training run consumes one budget unit. Therefore, an RS-SA-RE
repeat consumes real budget even though it does not create a new search
candidate.

Best-so-far search fitness updates only on a first evaluation that enters the
search population. A repeat evaluation has `inserted=False`; it leaves the
population and best-so-far search curve unchanged. Repeat accuracy is used only
to construct a paired instability label.

## Matched initialization

For each matched search seed, the first 20 initialization records must match in
both architecture and training seed:

| Search seed | Comparison | Architecture | Training seed | Result |
|---:|---|---:|---:|---|
| 2701 | RE vs SA-RE | 20/20 | 20/20 | PASS |
| 2701 | RE vs RS-SA-RE | 20/20 | 20/20 | PASS |
| 2702 | RE vs SA-RE | 20/20 | 20/20 | PASS |
| 2702 | RE vs RS-SA-RE | 20/20 | 20/20 | PASS |

Exact accuracy equality is deliberately not part of this audit. Architecture
and training-seed identity establish matched initialization; small accuracy
differences may still occur because of nondeterministic GPU execution.

The machine-readable evidence is stored in
`experiments/pilot/pilot_comparison_audit.json` and can be regenerated with:

```powershell
python scripts/audit_three_method_pilots.py `
  --pilot-root experiments/pilot
```

## Pilot-stage conclusion

The two matched seeds satisfy the architecture and training-seed initialization
requirements. The shared search-space, trainer, mutation, population,
tournament, aging, and real-budget conditions are fixed. The remaining
differences are the intended candidate-selection and repeat-label mechanisms.

