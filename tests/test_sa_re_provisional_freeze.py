"""Guards against accidental SA-RE pilot drift during the provisional freeze."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    2701: PROJECT_ROOT / "configs" / "pilot" / "sa_re_2701.yaml",
    2702: PROJECT_ROOT / "configs" / "pilot" / "sa_re_2702.yaml",
}
ALLOWED_RUN_IDENTITY_FIELDS = {"name", "search_seed", "output_dir"}


def _load_configs() -> dict[int, dict[str, object]]:
    configs: dict[int, dict[str, object]] = {}
    for seed, path in CONFIG_PATHS.items():
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert isinstance(config, dict), f"{path} must contain a YAML mapping"
        configs[seed] = config
    return configs


def _without_run_identity(config: dict[str, object]) -> dict[str, object]:
    normalized = deepcopy(config)
    experiment = normalized["experiment"]
    assert isinstance(experiment, dict)
    for field in ALLOWED_RUN_IDENTITY_FIELDS:
        experiment.pop(field)
    return normalized


class SAReProvisionalFreezeTests(unittest.TestCase):
    def test_frozen_pilot_configs_only_differ_by_run_identity(self) -> None:
        configs = _load_configs()
        self.assertEqual(
            _without_run_identity(configs[2701]),
            _without_run_identity(configs[2702]),
        )

    def test_frozen_search_and_surrogate_values(self) -> None:
        for config in _load_configs().values():
            evolution = config["evolution"]
            surrogate = config["surrogate"]
            self.assertIsInstance(evolution, dict)
            self.assertIsInstance(surrogate, dict)

            self.assertEqual(evolution["population_size"], 20)
            self.assertEqual(evolution["tournament_size"], 5)
            self.assertEqual(evolution["budget"], 30)
            self.assertEqual(evolution["candidate_count"], 5)

            self.assertEqual(surrogate["input_dim"], 280)
            self.assertEqual(surrogate["hidden_dims"], [32, 16])
            self.assertEqual(surrogate["optimizer"], "Adam")
            self.assertEqual(surrogate["learning_rate"], 0.001)
            self.assertEqual(surrogate["weight_decay"], 0.0001)
            self.assertEqual(surrogate["steps"], 200)
            self.assertEqual(surrogate["seed_offset"], 900000)

    def test_frozen_shared_training_protocol(self) -> None:
        for config in _load_configs().values():
            dataset = config["dataset"]
            network = config["network"]
            training = config["training"]
            self.assertIsInstance(dataset, dict)
            self.assertIsInstance(network, dict)
            self.assertIsInstance(training, dict)

            self.assertEqual(dataset["name"], "CIFAR10")
            self.assertEqual(dataset["split_seed"], 20260823)
            self.assertEqual(dataset["train_size"], 45000)
            self.assertEqual(dataset["val_size"], 5000)

            self.assertEqual(network["N"], 3)
            self.assertEqual(network["F"], 24)
            self.assertEqual(network["num_classes"], 10)

            self.assertEqual(training["epochs"], 5)
            self.assertEqual(training["batch_size"], 128)
            self.assertEqual(training["optimizer"], "SGD")
            self.assertEqual(training["training_seed_base"], 20260827)

    def test_frozen_run_identity_and_surrogate_seed_schedule(self) -> None:
        configs = _load_configs()
        expected_surrogate_seeds = {2701: 902701, 2702: 902702}
        for seed, config in configs.items():
            experiment = config["experiment"]
            surrogate = config["surrogate"]
            self.assertIsInstance(experiment, dict)
            self.assertIsInstance(surrogate, dict)

            self.assertEqual(experiment["method"], "SA-RE")
            self.assertEqual(experiment["mode"], "pilot")
            self.assertEqual(experiment["search_seed"], seed)
            self.assertEqual(
                experiment["output_dir"], f"experiments/pilot/sa_re_{seed}"
            )
            self.assertEqual(
                surrogate["seed_offset"] + seed,
                expected_surrogate_seeds[seed],
            )


if __name__ == "__main__":
    unittest.main()
