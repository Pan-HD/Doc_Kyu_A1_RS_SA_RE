# Method Fairness Audit

**Audit date:** 2026-08-28  
**Methods:** Regularized Evolution (RE) and Surrogate-Assisted Regularized Evolution (SA-RE)  
**Pilot search seeds:** 2701 and 2702

## Fairness table

| Item | RE | SA-RE | Fairness status |
|---|---:|---:|:---:|
| NASNet search space | same | same | ✅ |
| Network depth/filters (`N/F`) | `3/24` | `3/24` | ✅ |
| Training epochs | `5` | `5` | ✅ |
| Data split | same | same | ✅ |
| CNN trainer | same | same | ✅ |
| Fitness | final validation accuracy | final validation accuracy | ✅ |
| Population size (`P`) | `20` | `20` | ✅ |
| Tournament size (`S`) | `5` | `5` | ✅ |
| Tournament sampling | with replacement | with replacement | ✅ |
| Aging | FIFO | FIFO | ✅ |
| Mutation distribution | same | same | ✅ |
| Real CNN-training budget (`B`) | `30` | `30` | ✅ |
| Candidates per evolution step | `1` | `5` | intended difference |
| Child selection | direct mutation result | maximum surrogate-predicted mean | intended difference |

## Interpretation

The comparison holds the search space, data, network, training protocol, fitness, population, tournament selection, mutation process, aging rule, and real CNN-training budget fixed. The intended treatment difference is the offspring proposal and selection mechanism: RE evaluates one mutated child directly, whereas SA-RE generates five mutations and uses the surrogate to select one child for real training.

The pilot audit therefore passes on implementation fairness and budget conformance; it does not require SA-RE to achieve a higher final accuracy than RE. With only two matched seeds and a budget of 30, these runs demonstrate plausible search behavior but do not support a statistical superiority claim.

## Extension note

When RS-SA-RE is implemented, add an `RS-SA-RE` method column and apply the same checks. Any additional intended algorithmic difference must be identified explicitly rather than marked as a fairness mismatch.
