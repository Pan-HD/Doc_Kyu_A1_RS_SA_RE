import json
import random

from src.nasnet.genotype import (
    NASNetArchitecture,
    random_architecture,
    validate_architecture,
)

rng = random.Random(
    20260824
)

arch = random_architecture(
    rng
)

assert validate_architecture(
    arch
)

print(
    "Architecture valid:",
    validate_architecture(arch)
)

print(
    "Hash:",
    hash(arch)
)

serialized = json.dumps(
    arch.to_dict(),
    indent=2
)

print(serialized)

restored = (
    NASNetArchitecture.from_dict(
        json.loads(serialized)
    )
)

assert restored == arch

print(
    "Serialization round trip: PASSED"
)