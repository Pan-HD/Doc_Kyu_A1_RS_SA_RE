# Baseline Fidelity Record

## Source hierarchy

- **RE control flow:** Google Research official Regularized Evolution notebook + Real et al. (2019).
- **NASNet search space:** Real et al. (2019).
- **Tensor-level NASNet implementation details:** NASNet paper / reference implementation, with any implementation choices documented explicitly.

## Faithful algorithmic behavior

- tournament sampling: **with replacement**
- parent selection: highest fitness in sampled tournament
- one mutation event per evolutionary cycle
- child appended to population
- oldest individual removed (FIFO / aging)
- duplicate architectures are not automatically rejected

## Reduced-compute adaptations

| Item | Published setup | A1 preliminary setup |
|---|---:|---:|
| Population size `P` | 100 | 20 |
| Tournament size `S` | 25 | 5 |
| Search training epochs | 25 | 5 |
| Search evaluations | 20,000 | `B = 60` |

## Mutation fidelity target

- hidden-state mutation
- operation mutation
- identity/no-op mutation
- `p(identity) = 0.05`

## Baseline claim

> Faithful implementation of the Regularized Evolution algorithm in the NASNet search space under a reduced compute budget.

Do **not** describe this experiment as an exact full-scale reproduction of Real et al. (2019).

## Primary source links

- Real et al. (2019), *Regularized Evolution for Image Classifier Architecture Search*, AAAI 2019. DOI: 10.1609/aaai.v33i01.33014780
- Google Research official notebook: `evolution/regularized_evolution_algorithm/regularized_evolution.ipynb`

## Local archive note

`google_re_regularized_evolution.ipynb` in this folder is a lightweight reference-pointer notebook linking to the official Google Research upstream notebook; use the upstream link when checking exact source code.
