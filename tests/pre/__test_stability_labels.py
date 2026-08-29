from dataclasses import dataclass

import pytest

from src.surrogate.multitask_dataset import (
    PairedEvaluationRecord,
    PairedEvaluationStore,
)


@dataclass(frozen=True)
class DummyArchitecture:
    genotype: str


def test_single_seed_record_targets() -> None:
    record = PairedEvaluationRecord(
        base_evaluation_index=10,
        architecture=DummyArchitecture("X"),
        seed_1=101,
        accuracy_1=0.70,
    )

    assert record.has_pair is False
    assert record.mean_target == pytest.approx(0.70)
    assert record.instability_target is None


def test_paired_record_targets() -> None:
    record = PairedEvaluationRecord(
        base_evaluation_index=10,
        architecture=DummyArchitecture("X"),
        seed_1=101,
        accuracy_1=0.70,
    )
    record.add_repeat(seed_2=202, accuracy_2=0.76)

    assert record.has_pair is True
    assert record.mean_target == pytest.approx(0.73)
    assert record.instability_target == pytest.approx(0.06)


def test_duplicate_architectures_remain_distinct_records() -> None:
    architecture = DummyArchitecture("same-architecture")
    record_a = PairedEvaluationRecord(
        base_evaluation_index=10,
        architecture=architecture,
        seed_1=110,
        accuracy_1=0.70,
    )
    record_b = PairedEvaluationRecord(
        base_evaluation_index=25,
        architecture=architecture,
        seed_1=125,
        accuracy_1=0.72,
    )
    store: PairedEvaluationStore[DummyArchitecture] = PairedEvaluationStore()
    store.add(record_a)
    store.add(record_b)

    assert record_a.architecture == record_b.architecture
    assert record_a != record_b
    assert len(store) == 2
    assert store.get(10) is record_a
    assert store.get(25) is record_b


def test_repeat_seed_must_differ_from_first_seed() -> None:
    record = PairedEvaluationRecord(
        base_evaluation_index=10,
        architecture=DummyArchitecture("X"),
        seed_1=101,
        accuracy_1=0.70,
    )

    with pytest.raises(ValueError, match="differ"):
        record.add_repeat(seed_2=101, accuracy_2=0.76)


def test_record_cannot_receive_two_scheduled_repeats() -> None:
    record = PairedEvaluationRecord(
        base_evaluation_index=10,
        architecture=DummyArchitecture("X"),
        seed_1=101,
        accuracy_1=0.70,
    )
    record.add_repeat(seed_2=202, accuracy_2=0.76)

    with pytest.raises(ValueError, match="already"):
        record.add_repeat(seed_2=303, accuracy_2=0.74)


def test_store_rejects_duplicate_base_evaluation_index() -> None:
    store: PairedEvaluationStore[DummyArchitecture] = PairedEvaluationStore()
    store.add(
        PairedEvaluationRecord(
            base_evaluation_index=10,
            architecture=DummyArchitecture("X"),
            seed_1=101,
            accuracy_1=0.70,
        )
    )

    with pytest.raises(ValueError, match="duplicate base_evaluation_index"):
        store.add(
            PairedEvaluationRecord(
                base_evaluation_index=10,
                architecture=DummyArchitecture("Y"),
                seed_1=202,
                accuracy_1=0.72,
            )
        )

