import pytest
from sciSOM.SOM_recall.recall import *
from hypothesis import given, example
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
import numpy as np

array_strategy = st.lists(
    st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=100
).map(np.array)

def test_assign_labels():
    pass

# Make this more through by testing differnt size arrays
@given(arrays(np.float16, (3, 2), elements=st.floats(-1, 1)))
def test_affine_transform(data):
    data_max = np.max(data, axis=1)
    data_min = np.min(data, axis=1)

    if np.any(data_max == data_min):
        with pytest.raises(ZeroDivisionError):
            affine_transform(data, target_min = 0, target_max = 1)

    else:
        normalized_data = affine_transform(data, target_min = 0, target_max = 1)
        assert np.all(normalized_data >= 0) and np.all(normalized_data <= 1)
        assert np.max(normalized_data) == 1 and np.min(normalized_data) == 0

        normalized_data = affine_transform(data, target_min = -1, target_max = 1)
        assert np.all(normalized_data >= -1) and np.all(normalized_data <= 1)
        assert np.max(normalized_data) == 1 and np.min(normalized_data) == -1

# Need a test reference image to test the rest of the functions