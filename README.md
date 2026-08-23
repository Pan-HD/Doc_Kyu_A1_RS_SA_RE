# A1_RS_SA_RE

Experimental framework for:

- RE: Regularized / Aging Evolution baseline
- SA-RE: Surrogate-Assisted Regularized Evolution
- RS-SA-RE: Retraining-Stability-Aware Surrogate-Assisted Regularized Evolution

Current stage: **2026-08-23 experimental infrastructure + RE baseline**.

## Formal v0.3 search space

- `num_conv_blocks ∈ {2,3,4}`
- `initial_channels ∈ {16,24,32}`
- `channel_multiplier ∈ {1,2}`
- `kernel_size ∈ {3,5}`
- `dropout ∈ {0.0,0.25,0.5}`
- `use_batchnorm ∈ {False,True}`
- `activation ∈ {relu,gelu}`
- `pooling ∈ {max,avg}`

Total: **1,728 architectures**.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt

python scripts/check_search_space.py
pytest
python scripts/test_single_architecture.py
python scripts/run_re.py --config configs/debug.yaml
```

`data/raw/` is intentionally excluded from version control.
CIFAR-10 will be downloaded by torchvision when running the training scripts.

## Important

The official CIFAR-10 test set must not be used during architecture search.
All methods must reuse the same saved train/validation split.
