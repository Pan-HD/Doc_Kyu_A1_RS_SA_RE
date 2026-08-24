import random

from src.nasnet.genotype import (
    random_architecture,
    validate_architecture,
)

rng = random.Random(20260824)

arch = random_architecture(rng)

print(arch)

print(
    validate_architecture(arch)
)

print(
    hash(arch)
)

for i, pair in enumerate(
    arch.normal.pairs
):

    print(
        i,
        pair.branch_1.input_state,
        pair.branch_2.input_state
    )

import json

d = arch.to_dict()

text = json.dumps(
    d,
    indent=2
)

print(text)

from src.nasnet.genotype import (
    NASNetArchitecture,
)

arch2 = (
    NASNetArchitecture.from_dict(
        json.loads(text)
    )
)

print(
    arch == arch2
)

print(
    hash(arch) == hash(arch2)
)