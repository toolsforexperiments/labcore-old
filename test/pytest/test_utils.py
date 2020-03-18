import pytest
import numpy as np

from labcore.utils import same_type


def test_same_type():
    """test the type equality function"""

    for seq, exp in [
        ((1, ), True),
        ((1, 1.0, 1+0j), False),
        ((1, 10, 100), True),
        ((True, False, 10), False),
        (['abc', 'def', 'ghi'], True),
        (np.linspace(0, 1, 100), True),
    ]:
        assert same_type(*seq) == exp

    assert same_type(
        True, False, True, False, target_type=bool,
    )

    assert not same_type(
        True, False, True, False, target_type=int,
    )

    with pytest.raises(ValueError):
        same_type()
