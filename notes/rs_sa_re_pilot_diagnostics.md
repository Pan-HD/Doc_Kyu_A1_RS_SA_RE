# RS-SA-RE Pilot Diagnostics

## Scope and data integrity

This is an offline analysis of the frozen RS-SA-RE pilots for matched search seeds 2701 and 2702. No CNN training was run. The analysis contains 5 paired repeat labels per seed (10 combined) and 5 candidate-selection steps per seed (10 combined).

`instability_target = abs(accuracy_seed_1 - accuracy_seed_2)` and `mean_target = (accuracy_seed_1 + accuracy_seed_2) / 2`.

## True instability-label statistics

| Group | n | Mean | Median | Sample SD | Min | Max | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2701 | 5 | 0.009200 | 0.010200 | 0.007089 | 0.000400 | 0.019400 | 0.005600 |
| 2702 | 5 | 0.008680 | 0.007000 | 0.007314 | 0.000200 | 0.018000 | 0.010200 |
| combined | 10 | 0.008940 | 0.008600 | 0.006796 | 0.000200 | 0.019400 | 0.009050 |

## Surrogate scale diagnostic

Across the 10 candidate sets, the true instability targets range from 0.000200 to 0.019400. Candidate-set mean predicted instability ranges from 0.016033 to 0.116878.

For each K=5 set, the ranking-force ratio is `R = lambda * range(d_hat) / (range(mu_hat) + epsilon)`, with `lambda=1` and `epsilon=1e-12`. Observed R ranges from 0.135375 to 1.462141, with median 0.573147.

A small training-set instability MSE does not demonstrate calibrated candidate predictions. Only 4–5 paired labels are available at a selection step, and the candidate architectures are out-of-sample surrogate inputs.

## Offline lambda sensitivity

| Group | Lambda | Steps | Ranking changes | Frequency |
|---|---:|---:|---:|---:|
| 2701 | 0 | 5 | 0 | 0.000 |
| 2701 | 0.5 | 5 | 0 | 0.000 |
| 2701 | 1 | 5 | 0 | 0.000 |
| 2701 | 2 | 5 | 1 | 0.200 |
| 2702 | 0 | 5 | 0 | 0.000 |
| 2702 | 0.5 | 5 | 1 | 0.200 |
| 2702 | 1 | 5 | 1 | 0.200 |
| 2702 | 2 | 5 | 2 | 0.400 |
| combined | 0 | 10 | 0 | 0.000 |
| combined | 0.5 | 10 | 1 | 0.100 |
| combined | 1 | 10 | 1 | 0.100 |
| combined | 2 | 10 | 3 | 0.300 |

`lambda=0` is an invariant check and must always select the same candidate as `argmax(mu_hat)`.

## Interpretation constraint

Offline lambda sensitivity is a mechanism diagnostic, not performance tuning. Only the selected candidate has a real CNN accuracy; the other four candidates in each set have no outcome ground truth. A ranking change therefore shows that the stability penalty has a nonzero selection effect, but it cannot show that one lambda gives better performance.

## Provisional decision

Unless the CSV diagnostics reveal numerical collapse, NaN/Inf values, or complete domination by the instability prediction, retain `lambda=1.0` provisionally until the 8/31 formal freeze. The justification is its unit interpretation: a predicted instability increase of 0.01 receives the same penalty as a predicted mean-accuracy decrease of 0.01. This is not an outcome-driven retuning decision.
