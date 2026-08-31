# A1 Formal Experimental Protocol Freeze

## Parts D–K: Method, Configuration, and Protocol Freeze

**Project:** A1 Full NASNet — RE / SA-RE / RS-SA-RE  
**Freeze date:** 2026-08-31  
**Protocol status:** Methods and 30 formal configurations frozen  
**Formal execution status:** Not yet authorized; local Part L–N regression,
preflight, commit, and tag remain pending

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

This document also freezes the seed-specific configurations, RNG namespaces,
formal run manifest schema, duplicate handling, and real-training budget
definition. The final preflight, Git commit, and `A1-formal-v1` tag remain later
gates in the freeze workflow.

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

## 10. Immutable formal configuration set

The formal set contains exactly 30 seed-specific configurations:

| Method | Paths | Count |
|---|---|---:|
| RE | `configs/formal/re/re_1001.yaml` through `re_1010.yaml` | 10 |
| SA-RE | `configs/formal/sa_re/sa_re_1001.yaml` through `sa_re_1010.yaml` | 10 |
| RS-SA-RE | `configs/formal/rs_sa_re/rs_sa_re_1001.yaml` through `rs_sa_re_1010.yaml` | 10 |

Every file has a concrete search seed and output directory; a formal run must
not be created by editing a seed at launch time. All files have:

```text
experiment.mode       = formal
experiment.status     = frozen
experiment.do_not_run = false
experiment.overwrite  = false
```

The root-level `re.yaml`, `sa_re.yaml`, `rs_sa_re.yaml`, and `seeds.yaml` are
legacy pre-freeze source templates. They are not members of the 30-run formal
set and must not be passed to a formal runner. The authoritative seed mapping
is `configs/formal/matched_seed_manifest.yaml`.

`scripts/generate_formal_configs.py --check` verifies exact generated content.
`--write` may create missing artifacts but deliberately refuses to overwrite a
differing frozen artifact.

---

## 11. Formal seeds and RNG namespaces

The matched search seeds are:

```text
1001, 1002, 1003, 1004, 1005,
1006, 1007, 1008, 1009, 1010
```

For each seed, RE, SA-RE, and RS-SA-RE use the same search seed. The initial
architecture sequence must therefore match across methods before any
method-specific mechanism is used.

| Namespace | Frozen policy |
|---|---|
| Search RNG | `search_seed`; identical across the matched method triplet |
| Training RNG | Base `20260827`; existing evaluator replica schedule |
| Surrogate RNG | `900000 + search_seed` |
| Repeat RNG | `1800000 + search_seed` |

These namespaces are independent. Surrogate fitting or inference must not
advance the search RNG. Repeat scheduling or selection must not advance the
search RNG. Replica one preserves the existing first-training seed schedule;
the repeat replica is deterministic and distinct.

---

## 12. Duplicate, tournament, and FIFO policies

Natural or identity duplicate architectures are not filtered. If selected for
a new first evaluation, the duplicate is evaluated as a distinct evaluation
record, consumes one unit of `B`, and follows the same insertion/FIFO policy as
any other selected child.

RS-SA-RE repeat evaluations are not natural-duplicate first evaluations. A
repeat is attached to one exact unpaired base-evaluation record. A base record
may be paired at most once, and natural duplicate architectures must not be
merged when repeat labels are created.

Tournament sampling is with replacement and parent comparison uses true
first-seed fitness. Population aging is FIFO by insertion/birth order. A repeat
does not enter the population, create a birth event, change age, or replace
first-seed fitness.

---

## 13. Exact real-training budget definition

**The following operations do not count toward `B`: mutation, candidate
generation, surrogate training, surrogate inference, scoring, and logging.**

**Every selected new-architecture CNN training counts as one real-training
unit. In RS-SA-RE, every second-seed CNN retraining also counts as one
real-training unit.**

The exact completed-run expectations are:

| Method | Total `B` | First evaluations | Repeat evaluations | Evolution children | Candidate rows |
|---|---:|---:|---:|---:|---:|
| RE | 60 | 60 | 0 | 40 | Not applicable |
| SA-RE | 60 | 60 | 0 | 40 | 200 |
| RS-SA-RE | 60 | 49 | 11 | 29 | 145 |

No method may stop at 59 or consume a 61st real CNN training. Candidate rows
and predictions are budget-neutral; only the selected candidate is trained.

---

## 14. Formal run manifest and state machine

`experiments/formal/manifest.csv` is the authoritative 30-row run ledger. It is
ordered by search seed, then RE / SA-RE / RS-SA-RE, and contains these fields:

```text
method, search_seed, config_path,
status, start_time, end_time,
output_directory,
real_training_runs, first_evaluations, repeat_evaluations,
exit_code, audit_status, notes
```

The only valid run states are:

```text
pending -> running -> completed -> audited
                  \-> failed
```

An interrupted run is `failed` until resume/recovery is explicitly started.
`completed` means the process finished and produced outputs; `audited` means
the method-specific count and integrity audit passed. Empty numeric cells in a
pending row mean “not yet observed” and must not be replaced by fabricated
zeros.

---

## 15. Output and launch contract

Each formal configuration freezes a unique output directory:

```text
experiments/formal/re_<seed>
experiments/formal/sa_re_<seed>
experiments/formal/rs_sa_re_<seed>
```

`overwrite=false` is mandatory. Before launch, the manifest row moves from
`pending` to `running` with a start timestamp. On process exit, record the end
timestamp and exit code. A successful process becomes `completed`; it becomes
`audited` only after the expected budgets, row counts, population invariants,
and output files pass the method audit.

Formal execution commands must reference the per-seed file directly, for
example:

```text
python scripts/run_re.py --config configs/formal/re/re_1001.yaml
```

Do not add `--search-seed` or edit the YAML at launch time.

---

## 16. Git identity and authorization gate

The final freeze commit must be recorded before the first formal launch:

| Field | Required value |
|---|---|
| Git commit | `PENDING_FINAL_PREFLIGHT_COMMIT` |
| Git tag | `A1-formal-v1` |
| Worktree | Clean at tag creation |

Replace the pending commit marker with the actual immutable commit hash in the
tagged protocol version. If changing the marker creates a new commit, record
the final commit that is actually tagged.

The Part L–N runner update accepts `debug`, `pilot`, and `formal` modes. Formal
RS-SA-RE requires `B=60`, seeds 1001–1010, a concrete output directory,
`overwrite=false`, and the exact 49-first/11-repeat audit expectations. Formal
search and repeat seeds cannot be overridden on the command line.

---

## 17. Current file-mutation boundary

Parts H–K create or edit only:

```text
configs/formal/re/*.yaml
configs/formal/sa_re/*.yaml
configs/formal/rs_sa_re/*.yaml
configs/formal/matched_seed_manifest.yaml
scripts/generate_formal_configs.py
tests/test_formal_configs.py
tests/test_formal_manifest.py
experiments/formal/manifest.csv
notes/formal_freeze_2026-08-31.md
```

The following remain untouched by this package:

```text
src/**
scripts/run_re.py
scripts/run_sa_re.py
scripts/run_rs_sa_re.py
configs/pilot/**
experiments/stability_diagnostic/**
```

---

## 18. Pending completion items

The following gates remain pending:

- run the generator check, Part L–N tests, full regression,
  matched-seed/RNG preflight, and method-specific dry audits;
- record the final Git commit hash and confirm a clean worktree;
- create and verify tag `A1-formal-v1`;
- authorize and start formal RE seed 1001.

---

## 19. Freeze declaration

The RE, SA-RE, and RS-SA-RE definitions, 30 seed-specific configurations,
matched-seed mapping, manifest schema, RNG namespaces, duplicate policy, and
budget definition in this document are frozen as of 2026-08-31. The Stability
Diagnostic decision is **Yellow / GO**. No formal run is authorized until every
remaining preflight item in Section 18 passes.

---

## 20. Part L formal output contract

Every completed formal output directory must contain:

| Method | Required files |
|---|---|
| RE | `config.yaml`, `evaluations.csv`, `history.json`, `run.log`, `best.json` |
| SA-RE | RE files plus `candidate_predictions.csv` |
| RS-SA-RE | SA-RE files plus `repeat_evaluations.csv` |

`scripts/run_formal.py` is the only recommended formal launcher. It validates
the immutable config against `manifest.csv`, refuses an existing output
directory, marks the row `running`, invokes the method runner without seed
overrides, captures stdout and stderr, and flushes plus `fsync`s every received
line. It then verifies the output contract and method-specific counts before
marking the run `audited`.

The launcher can materialize a missing frozen `config.yaml` and derive a
missing `best.json` from a recognized fitness column in `evaluations.csv`. It
does not invent evaluations or candidate/repeat rows. The underlying search
implementation remains responsible for writing and flushing each evaluation
record immediately; failure to produce durable CSV/history artifacts fails the
output audit.

Use a dry run before the first launch:

```text
python scripts/run_formal.py \
  --config configs/formal/re/re_1001.yaml \
  --dry-run
```

The actual first formal command is:

```text
python scripts/run_formal.py \
  --config configs/formal/re/re_1001.yaml
```

Do not invoke a formal runner with `--search-seed` or `--repeat-seed`.

---

## 21. Part M formal preflight

The final preflight is no-GPU except for imports and does not run an epoch-1 or
real `B=60` search. It checks the 30 immutable configs, shared dataset/trainer/
NASNet fields, method-specific mechanisms, matched seed/RNG namespaces, budget
expectations, run manifest, output-contract files, generator immutability, and
all ten RS-SA-RE formal configs through `--validate-only`.

Run:

```text
python -m pytest \
  tests/test_stability_diagnostic_analysis.py \
  tests/test_rs_sa_re_formal_budget.py \
  tests/test_formal_configs.py \
  tests/test_formal_manifest.py \
  tests/test_formal_output_contract.py \
  tests/test_formal_preflight.py \
  tests/test_formal_runner.py -v

python -m pytest tests -v

python scripts/check_formal_preflight.py
```

The preflight writes `experiments/formal/preflight_report.json`. Formal launch
authorization requires `FORMAL PREFLIGHT: PASS` and a completely passing full
regression.

---

## 22. Part N Git formal freeze

After all Part M checks pass:

```text
git status
git add .
git commit -m "Freeze A1 formal experimental protocol"
git status
git tag -a A1-formal-v1 -m "A1 formal experimental protocol"
git show A1-formal-v1 --stat
```

The worktree must be clean before tag creation. The authoritative frozen commit
is the commit referenced by `A1-formal-v1`; obtain it with:

```text
git rev-parse A1-formal-v1^{commit}
```

After the tag, changes to crash recovery, logging, resume logic, analysis, or
plotting require an explicit assessment of whether they alter an architecture
trajectory or training result. Changes to method definitions, hyperparameters,
RNG policies, dataset split, trainer, NASNet, mutation, or fitness are forbidden
unless a correctness bug invalidates the experiment. A trajectory-affecting
fix normally requires rerunning all affected formal runs.

---

## 23. Part L–N file boundary

This Part L–N package changes exactly:

```text
scripts/check_formal_preflight.py
scripts/check_rs_sa_re_smoke.py
scripts/run_formal.py
scripts/run_rs_sa_re.py
tests/test_formal_output_contract.py
tests/test_formal_preflight.py
tests/test_formal_runner.py
notes/formal_freeze_2026-08-31.md
```

It does not modify the frozen 30 formal configs, matched seed manifest, initial
30-row run manifest, pilot/stability outputs, NASNet definition, trainer,
mutation, surrogate, or evolutionary engine.
