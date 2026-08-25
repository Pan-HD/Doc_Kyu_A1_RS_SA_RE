import pytest
import torch

from src.nasnet.operations import (
    OPS,
    build_operation,
)


@pytest.mark.parametrize(
    "op_name",
    OPS,
)
def test_normal_operation_shape(
    op_name,
):
    x = torch.randn(
        4, 24, 32, 32
    )

    op = build_operation(
        op_name,
        channels=24,
        stride=1,
    )

    y = op(x)

    assert y.shape == (
        4, 24, 32, 32
    )

@pytest.mark.parametrize(
    "op_name",
    OPS,
)
def test_reduction_operation_shape(
    op_name,
):
    x = torch.randn(
        4, 24, 32, 32
    )

    op = build_operation(
        op_name,
        channels=24,
        stride=2,
    )

    y = op(x)

    assert y.shape == (
        4, 24, 16, 16
    )

def test_unsupported_operation_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Unsupported NASNet operation",
    ):
        build_operation(
            "unsupported_op",
            channels=24,
            stride=1,
        )