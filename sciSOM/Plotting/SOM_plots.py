import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Union

def plot_SOM_gird_neurons(weight_cube: np.ndarray) -> None:
    """
    This function take in a nunr file from NeuroScope and converts it into a useful format to us
    Then it uses the data in the nunr file to identify which data samples belong to each PE
    Finally it takes this data and plots it such that we can overlay any data we want.
    
    Parameters:
    -------------------
    weight_cube : np.ndarray
        Weight cube after an SOM has been trained
        
    Returns:
    ----------------
    None
        
    """
    
    xgrid, ygrid, data_dim = np.shape(weight_cube)
    
    # Plotting section
    fig, ax = plt.subplots(nrows=ygrid, ncols=xgrid, figsize=(5, 5))

    a = 1
    for i in np.arange(ygrid):
        for j in np.arange(xgrid):
            ax[j,i].plot(weight_cube[j,i,:])
            ax[i,j].axis('off')
            ax[i,j].set_xlim(0, data_dim)
            ax[i,j].set_ylim(0, 1)
            

def SOM_gird_avg_wavefrom_per_cell(input_data: np.ndarray, 
                                   nunr_file_input: str, 
                                   grid_x_dim: int, 
                                   grid_y_dim: int, 
                                   x_dim_data_cube: int, 
                                   output_img_name: str, 
                                   is_struct_array = True):
    """
    Generates image of the average waveform for each cell in the SOM grid.

    This function take in a nunr file from NeuroScope and converts it into a useful format to us
    Then it uses the data in the nunr file to identify which data samples belong to each PE
    Finally it takes this data and plots it such that we can overlay any data we want.

    Parameters
    ----------
    input_data : int
        waveforms (peaks, peaklets)
    nunr_file_input : str 
        text file output from neuroscope
    grid_x_dim : int   
        SOM x-dimension
    grid_y_dim: int  
        SOM y-dimension
    x_dim_data_cube : int 
        x-dimension of the input data cube for the SOM
    output_img_name : str      
        name of file to save the image to + path
    is_struct_array : bool 
        does the data need to be accessed as peaks['data']?
    """
    
    # Import nunr file and convert it into a useful format
    import io
    nunr_file = []
    with io.open(nunr_file_input, mode="r", encoding="utf-8") as f:
        next(f)
        #next(f)
        for line in f:
            nunr_file.append(line.split())
    
    PE_data = nunr_to_obj(nunr_file, x_dim_data_cube)
    
    # Plotting section
    xgrid = grid_x_dim
    ygrid = grid_y_dim
    fig, ax = plt.subplots(nrows=ygrid, ncols=xgrid, figsize=(20, 20))

    a = 1
    for i in np.arange(ygrid):
        for j in np.arange(xgrid):
            if PE_data[(i+1) + (j)*xgrid] != []:
                [_, data_dim] = np.shape(input_data)
                data = input_data[np.array(PE_data[(i+1) + (j)*xgrid])-1]
                if is_struct_array == True:
                    ax[39-j,i].plot(np.mean(data['data'], axis = 0), alpha = a, color = 'black')
                elif is_struct_array == False:
                    ax[39-j,i].plot(np.mean(data, axis = 0), alpha = a, color = 'black')
            if PE_data[(i+1) + (j)*xgrid] == []:
                if is_struct_array == True:
                    ax[39-j,i].plot(np.zeros(200), alpha = a, color = 'black')
                else:
                    ax[39-j,i].plot(np.zeros(data_dim), alpha = a, color = 'black')
            #kind = kind + 1
            #ax[i,j].set_xlabel('Sample #')
            if is_struct_array == True:
                ax[i,j].set_xlim(0, 200)
            else:
                ax[i,j].set_xlim(0, data_dim)
            ax[i,j].axis('off')

    fig.savefig(output_img_name, bbox_inches='tight')
    
def calculate_u_matrix(weight_cube: np.ndarray):
    """
    Calculate the distance (fences) for each adjacent neuron in an SOM.

    (Need to review this function, dont fully remember what is going on in the
    implementation)

    Parameters
    ----------
    weight_cube : np.ndarray
        The weight cube for the SOM

    Returns
    -------
    u_matrix : np.ndarray
        The distance matrix for neurons in the SOM
    """
    x, y, _ = weight_cube.shape
    u_matrix = np.zeros((x, y))

    for i in range(x):
        for j in range(y):
            neighbors = []
            if i > 0:
                neighbors.append(weight_cube[i-1, j])
            if i < x-1:
                neighbors.append(weight_cube[i+1, j])
            if j > 0:
                neighbors.append(weight_cube[i, j-1])
            if j < y-1:
                neighbors.append(weight_cube[i, j+1])

            distances = [np.linalg.norm(weight_cube[i, j] - neighbor) for neighbor in neighbors]
            u_matrix[i, j] = np.mean(distances)

    return u_matrix

def calculate_density_matrix(weight_cube: np.ndarray, 
                             u_matrix: np.ndarray, 
                             dataset: np.ndarray) -> np.ndarray:
    """
    Calculate density matrix for a given som weight cube and dataset.

    **This function is not working as intended, need to review it**
    It is not acutally using the information of the u_matrix

    Parameters
    ----------

    weight_cube : np.ndarray
        SOM weight cube
    u_matrix : np.ndarray
        output from calculate_u_matrix
    dataset:    
        Data in the same form given to the SOM as input for training

    Returns
    -------
    density_matrix : np.ndarray
        The density matrix for the given dataset
    """
    x, y = u_matrix.shape
    density_matrix = np.zeros((x, y))

    for data_point in dataset:
        distances = cdist(weight_cube.reshape(-1, weight_cube.shape[-1]), [data_point], metric='euclidean')
        #print(np.shape(weight_cube.reshape(-1, weight_cube.shape[-1])))
        #print(np.shape(data_point))
        bmus = np.argmin(distances)
        x_idx, y_idx = np.unravel_index(bmus, (x, y))
        density_matrix[x_idx, y_idx] += 1

    return density_matrix

def display_density_matrix(density_matrix: np.ndarray):
    """
    Display the density matrix as an image.

    Parameters
    ----------
    density_matrix : np.ndarray
        The density matrix to display
    """
    import matplotlib.pyplot as plt
    plt.imshow(density_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('Density Matrix')
    plt.show()
    
def rise_time_vs_area_SOM_clusters(data: np.ndarray, colors: Union[list, np.ndarray], 
                                   n_rows: int, n_cols: int):
    """
    Plots the rise time vs area for each cluster in the SOM.

    Takes in the data from peaklet level data using the SOM classification
    and outputs a grid of plots showing each cluster.

    Parameters
    ----------
    
    data : np.ndarray     
        strudtured array with XENONnT data of data type peaks or peaklet
    colors : list or np.ndarray   
        list of colors used by the SOM
    n_rows:   
        number of coulmns in grid with the plots 
    n_cols:   
        number of rows in grid with the plots 
    """
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(24, 18))

    # Generalize this later
    colors = np.vstack((colors, np.array([0,0,0]).reshape((1, 3))))
    num = 0
    for i in np.arange(n_rows):
        for j in np.arange(n_cols):
            ax[i,j].scatter(data['area'][data['type'] == num],
                            -data['area_decile_from_midpoint'][data['type'] == num][:,1], 
                            s=0.5, color = colors[num]/255, alpha = 1)
            ax[i,j].set_xscale('log')
            ax[i,j].set_yscale('log')
            ax[i,j].set_xlim(1,10000000)
            ax[i,j].set_ylim(10,100000)
            num = num + 1


def nunr_file_to_list(nunr_file: list):
    """
    Extracts the PE data from a nunr file and converts it into a list.

    Parameters
    ----------
    nunr_file : list
        Opened nunr file from NeuroScope, needs to be parced to extract the PE data
    """
    obj = {}
    for i in range(1, len(nunr_file) + 1):
        obj[i] = []

    for i in np.arange(len(nunr_file)):  # also length of obj
        skip = 3  # every 3rd value is a char we dont care about
        for j in np.arange(int(len(nunr_file[i][2:]) / 3)):
            if (skip % 3) == 0:
                if tuple(nunr_file[i][skip:skip + 2]) != ('152', '991'):
                    obj[i + 1].append(tuple(nunr_file[i][skip:skip + 2]))
                    skip = skip + 3

    return obj


def tuple_list_to_list(tuple_list, x_data_img):
    obj = {}
    for i in range(1, len(tuple_list) + 1):
        obj[i] = []

    for i in np.arange(len(tuple_list)):
        for j in np.arange(len(tuple_list[i + 1])):
            obj[i + 1].append(int(tuple_list[i + 1][j][0]) + ((int(tuple_list[i + 1][j][1]) - 1) * x_data_img))

    return obj

def nunr_to_obj(nunr_file, x_data_img):
    return tuple_list_to_list(nunr_file_to_list(nunr_file), x_data_img)
