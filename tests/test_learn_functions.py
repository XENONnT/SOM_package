import pytest
from sciSOM.SOM_recall import affine_transform 
from sciSOM.SOM_learn.train import SOM
from hypothesis import given, example
from hypothesis.extra.numpy import arrays 
import hypothesis.strategies as st
import numpy as np

consistent_shape_strategy = st.integers(min_value=2, max_value=100).map(lambda x: (x,))

#array_strategy = st.lists(
#    st.floats(allow_nan=False, allow_infinity=False), min_size=2, max_size=100
#).map(np.array)

dtype_kohonen = np.dtype([
    ('time', 'i8'),  # Unicode string of max length 10
    ('alpha', 'float'),    # 4-byte integer
    ('sigma', 'float'),  # 4-byte float
    ('max_radius', 'i8')
])

learning_parameters_decay = np.zeros(1, dtype=dtype_kohonen)
learning_parameters_decay[0] = (1, 0.5, 0.5, 3)

learning_parameters_schedule = np.zeros(3, dtype=dtype_kohonen)
learning_parameters_schedule[0] = (100, 0.5, 0.5, 3)
learning_parameters_schedule[1] = (300, 0.1, 0.1, 2)
learning_parameters_schedule[2] = (1000, 0.01, 0.01, 1)

# I should probably vary the size of n_iter
n_iter = 1000

# Strategy to generate structured arrays for parameters with a consistent shape
@st.composite
def param_strategy(draw):
    # Generate each field separately
    time = draw(st.integers(min_value=1, max_value=1000))
    alpha = draw(st.floats(min_value=0.001, max_value=1.0))
    sigma = draw(st.floats(min_value=0.001, max_value=1.0))
    max_radius = draw(st.integers(min_value=1, max_value=3))
    
    # Create the structured array
    params = np.array([(time, alpha, sigma, max_radius)], dtype=dtype_kohonen)
    
    return params

# Strategy to generate data arrays for training with a consistent shape
@st.composite
def data_strategy(draw, shape):
    num_arrays = draw(st.integers(min_value=2, max_value=10)) 
    return [draw(arrays(dtype=np.float64,
                       shape=shape,
                       elements=st.floats(min_value=0.0, max_value=1.0))) for _ in range(num_arrays)]

@st.composite
def multiple_consistent_arrays_strategy(draw, dtype, shape_strategy):
    # Draw the shape using the shape_strategy, ensuring all arrays have the same shape
    consistent_shape = draw(shape_strategy)
    
    # Generate the number of arrays, at least 2
    num_arrays = draw(st.integers(min_value=2, max_value=10))
    
    # Generate multiple arrays with the consistent shape
    array_list = [draw(arrays(dtype=dtype, shape=consistent_shape, elements=st.floats(0, 1))) for _ in range(num_arrays)]
    
    return array_list

def is_monotonically_decreasing(arr):
    return all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))

def n_iter_test():
    return 1000
# We will test the SOM class and all of its methods
# So we will need several instances for different things.

#@given(@given(st.lists(array_strategy, min_size=2, max_size=100)))
@pytest.mark.parametrize("learn_parm", learning_parameters_decay)
def test_SOM_init(learn_parm):
    #length, dim = np.shape(arrays)
    som_model = SOM(x_dim = 5, y_dim = 5, input_dim = 5, 
                    n_iter = 1000, learning_parameters=learn_parm)
    assert som_model.weight_cube.shape == (5, 5, 5)
    assert som_model.n_iter == 1000
    assert som_model.learning_parameters['alpha'] == learn_parm['alpha']
    assert som_model.learning_parameters['sigma'] == learn_parm['sigma']
    assert som_model.learning_parameters['max_radius'] == learn_parm['max_radius']

#@given(data=multiple_consistent_arrays_strategy(dtype=np.float64, shape_strategy=consistent_shape_strategy))
@given(shape=consistent_shape_strategy,
       params=param_strategy(),
       data=multiple_consistent_arrays_strategy(dtype=np.float64, shape_strategy=consistent_shape_strategy))
def test_SOM_kohonen_defualt(shape, params, data):
    _,dim =np.shape(data)
    som_model = SOM(x_dim = 5, y_dim = 5, input_dim = dim, n_iter = 1000, 
                    learning_parameters=params,
                    )
    
    
    #array_normalized = affine_transform(data, target_min = 0, target_max = 1)
    som_model.train(data)
    assert som_model.is_trained == True
    assert is_monotonically_decreasing(som_model.learning_rate_history) == True
    assert is_monotonically_decreasing(som_model.learning_radius_history) == True


#@given(st.lists(array_strategy, min_size=2, max_size=100))
def pause_test_SOM_kohonen_decay_modes(arrays):
    array_normalized = affine_transform(arrays, target_min = 0, target_max = 1)
    som_model = SOM(5, 5, 5, n_iter = 1000, 
                    learning_parameters=learning_parameters_decay,
                    decay_mode = 'linear'
                    )
    som_model.train(array_normalized)
    assert som_model.is_trained == True



