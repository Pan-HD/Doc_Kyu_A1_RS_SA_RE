import json
from pathlib import Path

import pandas as pd
import yaml


def save_config_copy(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def save_search_history(history, output_dir, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    json_rows = []

    for record in history:
        row = {
            "training_run_index": record.training_run_index,
            "training_seed": record.training_seed,
            "architecture": json.dumps(record.architecture.to_dict(), sort_keys=True),
            "parent_architecture": (
                json.dumps(record.parent_architecture.to_dict(), sort_keys=True)
                if record.parent_architecture is not None
                else None
            ),
            "validation_accuracy": record.validation_accuracy,
            "parameter_count": record.parameter_count,
            "training_time": record.training_time,
            "birth_order": record.birth_order,
        }
        if metadata:
            row.update(metadata)
        rows.append(row)

        jrow = dict(row)
        json_rows.append(jrow)

    pd.DataFrame(rows).to_csv(output_dir / "evaluations.csv", index=False)

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(json_rows, f, indent=2)
