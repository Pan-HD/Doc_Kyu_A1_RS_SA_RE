# SA-RE Matched Pilot Update — 2026-08-28

This archive is an incremental update for the existing
`D:\Pro_Doc_Kyu\A1_RS_SA_RE` project. Extract it into the project root while
preserving the directory structure.

## Included changes

- Adds the fixed five-epoch SA-RE pilot configs for seeds 2701 and 2702.
- Adds a strict RE/SA-RE matched-initialization audit script.
- Prevents surrogate scoring from consuming the dedicated search RNG.
- Strengthens candidate, budget, population, and CSV metadata invariants.
- Extends toy and mocked end-to-end tests.

The current project stores the implementation under `src/search/`, so this
archive updates `src/search/surrogate_assisted_evolution.py` rather than
creating a duplicate `src/evolution/` module.

## Configuration source of truth

The pilot configs intentionally match the completed RE pilot configuration:

```text
CIFAR-10 train/validation: 45000 / 5000
training_seed_base:        20260827
N / F:                     3 / 24
epochs:                    5
P / S / K / B:             20 / 5 / 5 / 30
surrogate:                  280 -> 32 -> 16 -> 1
```

Do not change the two SA-RE configs independently. Their only experimental
differences are `search_seed`, experiment name, and output directory.

## Before running pilots

```powershell
python -m pytest tests -v
git status
```

Optional config comparison:

```powershell
git diff --no-index `
  configs/pilot/sa_re_2701.yaml `
  configs/pilot/sa_re_2702.yaml
```

Only the seed/name/output-directory lines should differ.

## Run seed 2701

```powershell
python scripts/run_sa_re.py `
  --config configs/pilot/sa_re_2701.yaml
```

Expected completion checks:

```text
real evaluations: 30
final population: 20
candidate rows:   50
selected rows:    10
```

## Audit seed 2701 before starting 2702

```powershell
python scripts/audit_matched_pilot.py `
  --re experiments/pilot/re_2701/evaluations.csv `
  --sa-re experiments/pilot/sa_re_2701/evaluations.csv `
  --population-size 20 `
  --expected-search-seed 2701 `
  --json-output experiments/pilot/matched_initialization_2701.json
```

Architecture and training-seed mismatches are hard failures. Accuracy is
reported as a diagnostic by default. Add `--require-accuracy-match` only when
the GPU training protocol is expected to be exactly deterministic.

Do not start seed 2702 unless architecture and training-seed matches are both
20/20.

## Run and audit seed 2702

```powershell
python scripts/run_sa_re.py `
  --config configs/pilot/sa_re_2702.yaml

python scripts/audit_matched_pilot.py `
  --re experiments/pilot/re_2702/evaluations.csv `
  --sa-re experiments/pilot/sa_re_2702/evaluations.csv `
  --population-size 20 `
  --expected-search-seed 2702 `
  --json-output experiments/pilot/matched_initialization_2702.json
```

## Files that must not be overwritten

- `experiments/pilot/re_2701/`
- `experiments/pilot/re_2702/`
- formal seed configurations 1001–1010

The pilot configs use `overwrite: false`, so an existing SA-RE result will not
be replaced accidentally.
