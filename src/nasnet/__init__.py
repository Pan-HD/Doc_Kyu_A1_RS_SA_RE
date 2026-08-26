"""Public API for the NASNet search-space implementation."""

from .cell import FactorizedReduction, NASNetCell, Projection
from .encoding import (
    ARCHITECTURE_ENCODING_DIM,
    BRANCH_ENCODING_DIM,
    NUM_BRANCHES,
    encode_architecture,
)
from .genotype import (
    NASNetArchitecture,
    get_unused_states,
    random_architecture,
    validate_architecture,
)
from .mutation import (
    HIDDEN_STATE_MUTATION,
    HIDDEN_STATE_MUTATION_PROBABILITY,
    IDENTITY_MUTATION,
    IDENTITY_MUTATION_PROBABILITY,
    MUTATION_TYPES,
    MutationResult,
    OPERATION_MUTATION,
    OPERATION_MUTATION_PROBABILITY,
    mutate_architecture,
)
from .network import NASNetCIFAR, build_nasnet
from .operations import OPS, build_operation, make_bn


__all__ = [
    "ARCHITECTURE_ENCODING_DIM",
    "BRANCH_ENCODING_DIM",
    "FactorizedReduction",
    "HIDDEN_STATE_MUTATION",
    "HIDDEN_STATE_MUTATION_PROBABILITY",
    "IDENTITY_MUTATION",
    "IDENTITY_MUTATION_PROBABILITY",
    "MUTATION_TYPES",
    "MutationResult",
    "NASNetArchitecture",
    "NASNetCIFAR",
    "NASNetCell",
    "NUM_BRANCHES",
    "OPERATION_MUTATION",
    "OPERATION_MUTATION_PROBABILITY",
    "OPS",
    "Projection",
    "build_nasnet",
    "build_operation",
    "encode_architecture",
    "get_unused_states",
    "make_bn",
    "mutate_architecture",
    "random_architecture",
    "validate_architecture",
]
