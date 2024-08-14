# First steps

In order to view some of the basics uses we recommend checkout out the notebook in examples/Kohonen_SOM.ipynb. Here we will provide a condenced version of what you should do!

```
from sciSOM import *

data = # Import your data here
normalized_data = affine_transform(data, 0, 1)

# Generate a set of parameters to use
dtype = np.dtype([
    ('time', 'i8'),  # Unicode string of max length 10
    ('alpha', 'float'),    # 4-byte integer
    ('sigma', 'float'),  # 4-byte float
    ('max_radius', 'i8')
])

# Example parameters (In this case time wont be used)
parameters[0] = (1000, 0.3, 0.2, 3)

# Initialize an SOM model
som_model_simple = SOM(x_dim = 7, y_dim = 7,    
                   input_dim = 2,                 
                   n_iter=40000,
                   learning_parameters=parameters, 
                   mode = "online", 
                   )

# Finally train your model! 
som_model_simple.train(normalized data)
```

This is how you would train a model from scratch! You can now use the plotting functions and recall functions to analyze the result of your weightcube!

However you might have trained yout SOM in some other system and simply want to use some of our functionality to analyze your data. In this case you can always just import the weight cube and use it as our fucntions as designed to work with arbitrary SOM weight cubes (as long as you trained with square lattices).