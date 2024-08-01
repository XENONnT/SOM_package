import numpy as np

class SOM:
    __doc__ = """
    SOM class:
    SOM object that can be trained and used to classify data.
    One of the goals of this object will be to switch between different
    SOM implementations. For now, it will be a simple implementation and the 
    cSOM implementation.
    """

    def __init__(self, 
                 x_dim: int, 
                 y_dim: int, 
                 input_dim: int, 
                 n_iter: int, 
                 learning_parameters: np.ndarray, 
                 decay_type: str = "functional", 
                 som_type: str="Kohonen", 
                 mode: str="batch"):
        """
        Initialize the SOM object.

        Parameters:
        x_dim (int): The x dimension of the SOM.
        y_dim (int): The y dimension of the SOM.
        input_dim (int): The dimension of the input data.
        n_iter (int): The number of iterations to train the SOM.

        learning_parameters (structured array): 
        for Khononen SOM -> The learning rate and sigma for the SOM.
        for a cSOM -> The learning rate, beta, and gamma of the SOM.
        (The main difference between this paper and our implementation
        is that we also update the immidiate neighbors of the BMU)
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=23839

        decay_type (str): The type of decay the SOM should follow.
        som_type (str): The type of SOM to use. Current options are Kohonen or cSOM, this can be expanded.
        mode (str): The mode of the SOM. Either batch or online.
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_dim = input_dim
        self.n_iter = n_iter
        self.learning_parameters = learning_parameters
        self.decay_type = decay_type
        self.som_type = som_type
        self.mode = mode
        self.weight_cube = np.random.rand(x_dim, y_dim, input_dim)

        self.mode_methods = {
            'batch': self._train_batch,
            'online': self._train_online
        }
        
        if mode not in self.mode_methods:
            raise ValueError(f"Mode {mode} is not supported. Choose from {list(self.mode_methods.keys())}")


    def train(self, data):
        """
        Train the SOM object.

        Parameters:
        data (np.array): The data to train the SOM on.
        """
        # Check if the learning parameters are correct
        check_field_exists(self.learning_parameters, "alpha")

        # Decide the order of the input for traiing:
        train_method = self.mode_methods.get(self.mode)
        if train_method:
            # Maybe output an array with random indexes to train the SOM?
            data_shuffled_index = train_method(data, self.n_iter)

        # Train the SOM
        if self.som_type == "Kohonen":

        for i in range(self.n_iter):
            for d in data:
                bmu = self.find_bmu(d)
                self.update_weights(d, bmu, i)

    def Kohonen_SOM(self, index):
        """
        Train the SOM using the Kohonen algorithm.
        """
        raise NotImplementedError

    def decay(self):
        """
        Decides how the learning rate and other parameters will decrease over time
        """
        if self.decay_type == "exponential":
            self.learning_rate = self.learning_rate * np.exp(-i / self.n_iter)
            self.sigma = self.sigma * np.exp(-i / self.n_iter)

        elif self.decay_type == "linear":
            self.learning_rate = self.learning_rate - self.learning_rate / self.n_iter * i
            self.sigma = self.sigma - self.sigma / self.n_iter * i

        elif self.decay_type == "schedule":
            self

    def _train_batch(self, data, n_iter):
        """
        Train the SOM in batch mode.
        """
        raise NotImplementedError

    def _train_online(self, data, n_iter):
        """
        Train the SOM in online mode.
        """
        raise NotImplementedError



def check_field_exists(structured_array: np.ndarray, field_name: str) -> bool:
    """
    Check if a field exists in a structured array.
    """
    return field_name in structured_array.dtype.names

def log_parameters(parameters: dict):
    """
    Make a log of the parameters used to train the SOM and save this onto a file.
    """
    raise NotImplementedError