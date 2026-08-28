# SA-RE Provisional Freeze

**Status:** PROVISIONALLY FROZEN  
**Freeze period:** 2026-08-28 through 2026-08-31 (inclusive)  
**Canonical pilot seeds:** 2701 and 2702  
**Recommended Git tag:** `A1-sa-re-provisional-freeze-v1`

## Freeze decision

The two matched SA-RE pilots completed successfully and passed the initialization, budget, candidate-log, surrogate-diagnostic, comparison, and fairness audits. The SA-RE core definition below is therefore frozen until the scheduled review on 2026-08-31.

Pilot outcomes are not tuning signals. In particular, SA-RE seed 2701 finishing below RE seed 2701 is not a reason to change the surrogate architecture, candidate count, or optimization settings.

## Frozen specification

| Component | Frozen value |
|---|---|
| Population size | `P = 20` |
| Tournament size | `S = 5` |
| Candidate count | `K = 5` |
| Architecture encoding | `280-D` |
| Surrogate architecture | `280 → 32 → 16 → 1` |
| Surrogate output | one linear, unbounded scalar `mu_hat` |
| Target | final validation accuracy on the `0.0–1.0` scale |
| Loss | MSE |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Surrogate optimization steps | `200` |
| Candidate selection | `argmax(mu_hat)`; first candidate wins an exact tie |
| Parent selection | tournament on true measured fitness |
| Tournament sampling | with replacement |
| Aging | FIFO |
| Pilot real-training budget | `B = 30` |
| Budget accounting | only completed real CNN trainings consume budget |
| Real offspring evaluations | exactly one selected child per evolution step |
| Surrogate training data | all completed real evaluations only |
| Prediction-time dataset sizes | `20, 21, ..., 29` |
| Final surrogate dataset size | `30` |
| Surrogate seed schedule | `900000 + search_seed` |
| RNG isolation | search, CNN training, and surrogate RNG streams remain separate |

The canonical configuration artifacts during the freeze are:

```text
configs/pilot/sa_re_2701.yaml
configs/pilot/sa_re_2702.yaml
```

The two files must remain identical after removing only these run-identity fields:

```text
experiment.name
experiment.search_seed
experiment.output_dir
```

## Allowed freeze exceptions

Core SA-RE files may be changed before 2026-08-31 only to correct one of the following defects:

- non-finite values (`NaN` or infinity);
- demonstrated optimization divergence;
- encoding or tensor-dimension bug;
- seed schedule or RNG-isolation bug;
- real-budget accounting bug;
- logging defect that makes the audit invalid.

An exception requires reproducible evidence. Before changing code, record:

| Field | Required record |
|---|---|
| Date | discovery and fix date |
| Defect | exact failure and reproduction command |
| Evidence | failing test, log row, traceback, or audit result |
| Scope | affected source/configuration files |
| Change | before/after behavior |
| Artifact validity | whether existing 2701/2702 results remain valid |

If an exception changes architecture generation, parent selection, mutation, surrogate predictions, selected children, training targets, or budget consumption, do not overwrite the existing pilot directories. Version the fix, create new output directories, rerun both matched seeds, and repeat Parts B–D.

## Prohibited outcome-driven tuning

The following changes are prohibited during the freeze when motivated only by pilot accuracy:

- changing `K=5` to `K=10`;
- changing hidden layers from `[32, 16]` to `[64, 32]` or another shape;
- increasing surrogate steps from `200` to `1000`;
- changing loss, optimizer, learning rate, or weight decay;
- clipping or otherwise changing the surrogate output solely because a pilot lost;
- changing the candidate-selection rule after viewing RE versus SA-RE outcomes.

Such changes may be evaluated later only as a separately versioned ablation or method revision. They must not retroactively redefine the completed SA-RE pilots.

## Evidence retained with the freeze

- matched initial architectures: `20/20` for seeds 2701 and 2702;
- matched CNN training seeds: `20/20` for seeds 2701 and 2702;
- real CNN trainings: `30` per SA-RE pilot;
- candidate prediction rows: `50` per SA-RE pilot;
- selected candidate rows: `10` per SA-RE pilot;
- surrogate predictions: finite for both pilots;
- candidate screening: exactly one maximum-prediction candidate selected per step;
- Part C artifact-level comparison audit: `PASS`.

Exact validation-accuracy equality during matched initialization remains diagnostic rather than a hard gate because GPU training was not fully deterministic.

## Review on 2026-08-31

At the review, decide whether to:

1. retain the frozen SA-RE definition for the next experimental stage;
2. begin RS-SA-RE as a separate method;
3. define separately versioned surrogate ablations;
4. increase the number of search seeds or real-training budget.

No unfreeze is implied before that review.
