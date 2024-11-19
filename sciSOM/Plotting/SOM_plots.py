import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import Union
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
from ..SOM_recall.recall import SOM_location_recall


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

    plt.show()


def plot_mU_matrix(weight_cube: np.ndarray, 
                   data: np.ndarray,
                   set_costum_min_max: bool = False,
                   fence_vmin: float = None,
                   fence_vmax: float = None,
                   density_vmin: float = None,
                   density_vmax: float = None,
                  log_density: bool = False,
                  fence_on: bool = True):

    """
    Plots the mU-matrix; defined here as the data density per cell
    and the lines between cells representing the distance between
    adjacent cells.
    
    Parameters:
    -------------------
    weight_cube : np.ndarray
        Weight cube after an SOM has been trained
    data : np.ndarray
        Data used to train the SOM or data to be mapped to the SOM
    set_costum_min_max : bool
        If True, the user can set the vmin and vmax for fences
    fence_vmin : float
        Minimum value for the fences
    fence_vmax : float
        Maximum value for the fences
    density_vmin : float
        Minimum value for the density matrix (not implemented yet)
    density_vmax : float
        Maximum value for the density matrix (not implemented yet)
    log_density : bool
        If True applies a log to the density matrix calculation
    fence_on : bool
        If False removes fences from mU matrix image

    Returns:
    ----------------
    None
    
    """
    height, width, som_dim = np.shape(weight_cube)
    w_cube = weight_cube
    data_points, data_dim = np.shape(data)
    assert som_dim == data_dim
    
    cmap = LinearSegmentedColormap.from_list('black_to_red', ['black', 'red'])
    
    som_shape = (height, width)

    # Initialize grid to store counts of data points mapped to each node
    count_grid = np.zeros(som_shape)

    # Calculate the BMU (Best Matching Unit) for each data point
    for point in data:
        # Compute distances to each neuron
        distances = np.linalg.norm(weight_cube - point, axis=-1)
        # Find index of the neuron with the smallest distance
        bmu_index = np.unravel_index(np.argmin(distances), som_shape)
        count_grid[bmu_index] += 1

    # Normalize count_grid for color mapping
    if log_density == True:
        count_grid = np.log10(count_grid + 1)
        
    if set_costum_min_max is False:
        norm_counts = count_grid / np.max(count_grid)
    else:
        if density_vmax == None:
            norm_counts = count_grid / np.max(count_grid)
        else:
            norm_counts = count_grid / np.max(density_vmax)
    
    # In progress
    if fence_on:
        down_shifted_weight_cube = np.vstack((w_cube[-1:, :, :], 
                                               w_cube[:-1, :, :]))

        right_shifted_weight_cube = np.hstack((w_cube[:, -1:, :], 
                                             w_cube[:, :-1, :], ))

        vertical_lines = np.sqrt(np.sum((w_cube - down_shifted_weight_cube) ** 2, 
                                        axis=-1))
        horizontal_lines = np.sqrt(np.sum((w_cube - right_shifted_weight_cube) ** 2, 
                                        axis=-1))

        # Need to drop first row/column since its comparing opposite edges
        if set_costum_min_max == False:
            vmin = min(np.min(vertical_lines[1:,:]), np.min(horizontal_lines[:,1:]))
            vmax = max(np.max(vertical_lines[1:,:]), np.min(horizontal_lines[:,1:]))
        elif set_costum_min_max == True:
            if ((fence_vmin or fence_vmax) == None):
                vmin = min(np.min(vertical_lines[1:,:]), np.min(horizontal_lines[:,1:]))
                vmax = max(np.max(vertical_lines[1:,:]), np.min(horizontal_lines[:,1:]))
            else:
                vmin = fence_vmin
                vmax = fence_vmax
    
    fig, ax = plt.subplots()
    
    for i in range(height):
        for j in range(width):
            ax.add_patch(plt.Rectangle((j, height - i - 1), 1, 1,
                                       color=cmap(norm_counts[i, j]),
                                       ec='black'))
    
    if fence_on:
        for i in range(height):
            for j in range(width):
                if i < height - 1:  # Vertical line (between current and below)
                    #u_diff = np.linalg.norm(weightcube[i, j] - weightcube[i + 1, j])
                    color = plt.cm.gray(vertical_lines[i+1,j] / vmax)
                    ax.plot([j, j + 1], [height - i - 1, height - i - 1], color=color)

                if j < width - 1:
                    color = plt.cm.gray(horizontal_lines[i,j+1] / vmax)
                    ax.plot([j + 1, j + 1], [height - i - 1, height - i], color=color)
            
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off the axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()    

    
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


def SOM_gird_avg_wavefrom_per_cell(input_data: np.ndarray, 
                                   weight_cube: np.ndarray, 
                                   output_img_name: str = 'avg_waveform.png', 
                                   save_fig: bool = False,
                                   is_struct_array: bool = True):
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
    
    # Plotting section
    xgrid, ygrid, dim = np.shape(weight_cube)
    datapoints, data_dim = np.shape(input_data)
    assert dim == data_dim
    
    # Need to assign a location for each tuple
    # Need to also take into account dead neurons
    location_info = SOM_location_recall(weight_cube, input_data)

    fig, ax = plt.subplots(nrows=ygrid, ncols=xgrid, figsize=(5, 5))

    a = 1
    # Modify this monstrosity to deal with the current data formate
    # Remember dead neurons !!!
    for i in range(ygrid):
        for j in range(xgrid):
            loc_data = input_data[np.all(location_info.T == [j,i], axis=1)]
            if loc_data.size > 0:
                if is_struct_array == True:
                    ax[i,j].plot(np.mean(loc_data['data'], axis = 0), alpha = a, color = 'black')
                elif is_struct_array == False:
                    ax[i,j].plot(np.mean(loc_data, axis = 0), alpha = a, color = 'black')
            else:
                # Maybe replace this with red X's?
                ax[i,j].plot(np.zeros(data_dim), alpha = a, color = 'red')

            #kind = kind + 1
            #ax[i,j].set_xlabel('Sample #')
            if is_struct_array == True:
                ax[i,j].set_xlim(0, data_dim)
                ax[i,j].set_ylim(0, 1)
            else:
                ax[i,j].set_xlim(0, data_dim)
                ax[i,j].set_ylim(0, 1)
            ax[i,j].axis('off')

    if save_fig == True:
        fig.savefig(output_img_name, bbox_inches='tight')


def SOM_location_recall(weight_cube: np.ndarray,
                        normalized_data: np.ndarray,) -> np.ndarray:
    """
    Takes the data, the weight cube and the classification map and assignes each
    data point a label based on their cluster.

    Parameters
    ----------
    array_to_fill : np.ndarray
        structured array to fill with the classification
    data_in_SOM_fmt : np.ndarray
        data to classify in the SOM format
    weight_cube : np.ndarray
        SOM weight cube
    reference_map : np.ndarray
        reference map for the SOM

    Returns
    -------
    array_to_fill : np.ndarray
        structured array with the SOM classification added
    """

    # Want to make it so it works with different metrics in the future
    #array_to_fill = np.empty((len(normalized_data), 2))
    [SOM_xdim, SOM_ydim, _] = weight_cube.shape
    distances = cdist(
        weight_cube.reshape(-1, weight_cube.shape[-1]), normalized_data, metric="euclidean"
    )
    w_neuron = np.argmin(distances, axis=0)
    x_idx, y_idx = np.unravel_index(w_neuron, (SOM_xdim, SOM_ydim))
    array_to_fill = np.vstack((y_idx, x_idx))
    return array_to_fill