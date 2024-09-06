import numpy as np
import matplotlib.pyplot as plt


def SOM_gird_avg_wavefrom_per_cell_ns(input_data: np.ndarray, 
                                   weight_cube: np.ndarray, 
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


def tuple_list_to_list(tuple_list: tuple, x_data_img: int):
    obj = {}
    for i in range(1, len(tuple_list) + 1):
        obj[i] = []

    for i in np.arange(len(tuple_list)):
        for j in np.arange(len(tuple_list[i + 1])):
            obj[i + 1].append(int(tuple_list[i + 1][j][0]) + ((int(tuple_list[i + 1][j][1]) - 1) * x_data_img))

    return obj

def nunr_to_obj(nunr_file: np.ndarray, x_data_img: int):
    return tuple_list_to_list(nunr_file_to_list(nunr_file), x_data_img)