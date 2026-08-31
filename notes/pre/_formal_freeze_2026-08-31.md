# A1 Formal Experimental Protocol Freeze

## Part D–F: Method Definition Freeze

**Project:** A1 Full NASNet — RE / SA-RE / RS-SA-RE  
**Freeze date:** 2026-08-31  
**Protocol status:** Method definitions frozen  
**Formal execution status:** Not yet authorized; Part H–N preflight remains pending

---

## 1. Purpose and scope

This document freezes the method definitions that will be used in the formal
comparison of Regularized Evolution (RE), Surrogate-Assisted Regularized
Evolution (SA-RE), and Retraining-Stability-Aware Surrogate-Assisted
Regularized Evolution (RS-SA-RE).

The freeze is made after completion of:

- RE, SA-RE, and RS-SA-RE pilot runs for search seeds 2701 and 2702;
- the three-method fairness audit;
- the RS-SA-RE formal-budget dry run at `B=60`;
- the 12-architecture × 3-seed stability diagnostic;
- the SD@5 versus SD@25 correlation analysis.

This document freezes the experimental definitions only. Formal configurations,
the matched-seed manifest, the formal run manifest, the final preflight, the Git
commit, and the `A1-formal-v1` tag are completed in later parts of the freeze
workflow.

---

## 2. Stability Diagnostic decision

### 2.1 Frozen diagnostic design

| Item | Frozen value |
|---|---:|
| Architectures | 12 |
| Seeds per architecture | 3 |
| Total real training runs | 36 |
| Training trajectory | One continuous 25-epoch trajectory per run |
| Recorded milestones | Accuracy at epochs 5 and 25 |
| SD definition | Sample standard deviation (`ddof=1`) |
| Architecture seed | 8300 |
| Training seeds | 83001, 83002, 83003 |

The diagnostic used the same three independent seeds for every architecture.
Acc@5 and Acc@25 were obtained from the same training trajectory rather than
from separate training runs.

### 2.2 Result

| Statistic | Value |
|---|---:|
| Number of architectures | 12 |
| Spearman’s rho | 0.405594 |
| Two-sided p-value | 0.190836 |
| Direction | Positive |
| Gate decision | **Yellow / GO** |

### 2.3 Interpretation

The diagnostic showed a moderate positive association between short-budget
and longer-budget retraining instability (`rho=0.406`, `p=0.191`, `n=12`).
Given the small diagnostic sample, this is limited but directionally supportive
evidence that 5-epoch instability contains a useful signal for stability-aware
screening. It is not conclusive statistical validation of the proxy.

The diagnostic therefore supports proceeding with the frozen RS-SA-RE method,
while the limited diagnostic sample and non-significant p-value must be reported
as limitations.

### 2.4 Decision constraints

- Do not resample the 12 diagnostic architectures.
- Do not repeat the diagnostic to search for a more favorable correlation.
- Do not select a different epoch milestone after observing the result.
- Do not tune `lambda` using the diagnostic or the 2701/2702 pilot winners.
- Preserve the diagnostic as a methodological check rather than a formal
  large-sample validation.

---

## 3. Common frozen protocol

The following fields must be identical across RE, SA-RE, and RS-SA-RE.

| Parameter | Formal value |
|---|---|
| Dataset | CIFAR-10 |
| Search training set | 45,000 examples |
| Validation set | 5,000 examples |
| Dataset split seed | 20260823 |
| NASNet `N` | 3 |
| NASNet initial `F` | 24 |
| Architecture encoding | 280-D |
| Search training epochs | 5 |
| Batch size | 128 |
| Optimizer | SGD |
| Learning rate | 0.025 |
| Momentum | 0.9 |
| Weight decay | `5e-4` |
| AMP | On |
| Fitness | Final validation accuracy |
| Population size `P` | 20 |
| Tournament size `S` | 5 |
| Real-training budget `B` | 60 |
| Tournament sampling | With replacement |
| Aging policy | FIFO |
| Mutation operator and distribution | Identical frozen implementation |
| Parent fitness | True first-seed fitness |

The following do not consume the real-training budget:

- mutation;
- candidate generation;
- surrogate training;
- surrogate inference;
- scoring and logging.

Every new-architecture CNN training consumes one unit of `B`. In RS-SA-RE,
every second-seed CNN repeat evaluation also consumes one unit of `B`.

---

## 4. RE frozen specification

RE is the published-baseline implementation used as the common evolutionary
core.

| Item | Frozen value |
|---|---|
| Extra candidates per evolutionary step | 1 |
| Surrogate | None |
| Candidate choice | Direct evaluation |
| Parent selection | Tournament with replacement |
| Parent comparison | True first-seed fitness |
| Aging | FIFO |
| Repeat evaluations | None |

RE uses the same NASNet space, trainer, mutation operator, fitness, population,
tournament, aging policy, and total real-training budget as the other methods.

---

## 5. SA-RE frozen specification

| Item | Frozen value |
|---|---|
| Candidate count `K` | 5 |
| Encoding | 280-D |
| Surrogate | `280 -> 32 -> 16 -> 1` |
| Prediction target | Accuracy / mean-performance estimate `mu_hat` |
| Hidden activation | ReLU |
| Loss | MSE |
| Surrogate optimizer | Adam |
| Surrogate learning rate | `1e-3` |
| Surrogate weight decay | `1e-4` |
| Surrogate training steps | 200 |
| Candidate selection | `argmax mu_hat` |
| Parent selection | True first-seed fitness |
| Repeat evaluations | None |

Each evolutionary step generates five mutation candidates. The surrogate scores
all five candidates, but only the candidate with the highest predicted mean is
trained with the real CNN evaluator. Candidate generation, surrogate training,
and surrogate prediction do not consume `B`.

The pilot architecture `280 -> 32 -> 16 -> 1` is frozen. It must not be changed
to a larger model after observing formal results.

---

## 6. RS-SA-RE frozen specification

### 6.1 Multi-task surrogate

| Item | Frozen value |
|---|---|
| Candidate count `K` | 5 |
| Encoding | 280-D |
| Shared trunk | `280 -> 32 -> 16` |
| Heads | Mean head `mu_hat`; instability head `d_hat` |
| Hidden activation | ReLU |
| Mean target | First-seed accuracy, or two-seed mean when paired |
| Instability target | `d = abs(y1 - y2)` for paired evaluations |
| Loss | Mean MSE + masked instability MSE |
| Loss weight | 1.0 for each task |
| Surrogate optimizer | Adam |
| Surrogate learning rate | `1e-3` |
| Surrogate weight decay | `1e-4` |
| Surrogate training steps | 200 |

### 6.2 Candidate score

The frozen score is:

```text
Score(a) = mu_hat(a) - lambda * d_hat(a)
```

with:

```text
lambda = 1.0
```

The selected candidate is:

```text
argmax Score(a), over K=5 candidates
```

Parent selection continues to use true first-seed fitness. Predicted mean,
predicted instability, paired mean, or repeat accuracy must not replace the
population’s first-seed fitness.

### 6.3 Repeat policy

| Item | Frozen value |
|---|---:|
| Warm-up pairs | 4 |
| Repeat interval | Every 4 new first evaluations |
| Repeat selection | Fixed independent repeat-RNG policy |
| Duplicate pairing | One scheduled repeat per base-evaluation record |

Every repeat evaluation must satisfy all of the following:

- consumes one real CNN-training budget unit;
- uses the frozen second-seed policy;
- creates a paired stability label for the same base-evaluation record;
- does not create or insert a new population member;
- does not change population size;
- does not change FIFO birth order or age;
- does not replace first-seed population fitness;
- does not advance the matched search RNG stream.

---

## 7. Lambda freeze and unit justification

Accuracy and instability are both represented as accuracy fractions.

```text
Delta mu_hat = 0.01  -> predicted accuracy differs by 1 percentage point
Delta d_hat  = 0.01  -> predicted instability differs by 1 percentage point
```

Consequently, `lambda=1.0` places the predicted mean benefit and predicted
instability penalty on the same accuracy-fraction scale. The pilot diagnostics
also confirmed that the stability penalty can change candidate ranking; it is
not a dead term in the scoring expression.

The value is frozen as a unit-based methodological choice. It must not be
changed in response to the relative performance of pilot seeds 2701/2702 or
formal seeds 1001–1010.

---

## 8. Fairness boundary

Allowed method differences are limited to:

| Mechanism | RE | SA-RE | RS-SA-RE |
|---|---|---|---|
| Candidate count | 1 | 5 | 5 |
| Surrogate | None | Mean only | Mean + instability |
| Candidate score | Direct | `mu_hat` | `mu_hat - d_hat` |
| Repeat evaluations | None | None | Frozen repeat policy |

All other shared fields in Section 3 must remain identical.

---

## 9. Post-freeze change policy

### 9.1 Changes that are not allowed

After this freeze, do not change:

- `lambda`, `K`, `P`, `S`, or `B`;
- search epochs or trainer hyperparameters;
- surrogate architecture, losses, or optimization settings;
- warm-up pairs or repeat interval;
- dataset split;
- NASNet definition or encoding;
- tournament, FIFO, mutation, or fitness definitions;
- training-seed, search-seed, repeat-seed, or surrogate-seed policies.

A correctness bug that invalidates completed formal runs is the only exception.
Such a bug must be documented, and affected formal runs normally must be rerun.

### 9.2 Changes that may be allowed after review

The following may be changed only when they do not alter an architecture
trajectory or training result:

- crash recovery;
- resume handling;
- logging and metadata formatting;
- audit scripts;
- analysis scripts and plots.

---

## 10. Current file-mutation boundary

At the completion of Part D–F, the intended project mutation is limited to this
protocol note:

```text
CREATE/EDIT:
notes/formal_freeze_2026-08-31.md

DO NOT EDIT YET:
configs/formal/**
tests/test_formal_configs.py
src/**
experiments/stability_diagnostic/**
```

The formal configurations remain pre-freeze and non-runnable until Part H.
This method-definition freeze does not itself authorize formal seed 1001.

---

## 11. Pending completion items

The following items are intentionally pending:

- formal `B=60` protocol and exact count table in Part G;
- 30 immutable formal configs for seeds 1001–1010 in Part H;
- final formal-config tests in Part I;
- `experiments/formal/manifest.csv` in Part J;
- complete RNG, duplicate, output, and budget definitions in Part K;
- formal preflight and full regression;
- Git commit hash and clean-worktree confirmation;
- Git tag `A1-formal-v1`;
- authorization to start formal RE seed 1001.

---

## 12. Freeze declaration

The RE, SA-RE, and RS-SA-RE method definitions in this document are frozen as
of 2026-08-31. The Stability Diagnostic decision is **Yellow / GO**. The project
may proceed to formal-protocol finalization, but no formal run is authorized
until all remaining preflight requirements pass.

