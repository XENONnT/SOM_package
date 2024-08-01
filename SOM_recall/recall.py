import numpy as np
from typing import Any, Union, Dict
#import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy.lib.recfunctions as rfn
import strax
import straxen
import viff

# Lets organize this a bit better
# Need to move image manipulation functions to a separate file
def assign_labels(data: np.ndarray, ref_img: np.ndarray, xdim: int, ydim: int, cut_out) -> Any:
    '''This functions takes in the data and classifications based on an image gives the
    unique labels as well as the data set bacl with the new classification
    PS this version only takes in S1s and S2s and ignores unclassified samples, 
    another version will be made to deal with the unclassified samples
    
    data: can be either peaks or peak_basics
    ref_img: will be the image extracted from the SOM classification of each data point
    xdim: width of the image cube
    ydim: height of the image cube
    cut_out: removes the n last digits of the image vector if necesarry'''

    # This relays on reference images which is a NeuroScope implementation
    # 

    from PIL import Image
    data_new = data
    img = Image.open(ref_img)
    imgGray = img.convert('L')
    img_color = np.array(img) #still in the x,y,3 format
    img_color_2d = img_color.reshape((xdim*ydim,3))
    label = -1 * np.ones(img_color.shape[:-1])
    colorp = np.unique(img_color_2d, axis = 0)
    for i, color in enumerate(colorp):  # argwhere
        label[np.all((img_color == color), axis = 2)] = i #assignes each color a number
    label_vec = label.reshape((xdim*ydim))
    if cut_out != 0:
        label_vec_nonzero = label_vec[:-cut_out]
    elif cut_out == 0:
        label_vec_nonzero = label_vec
    #s2_data = data[data['type'] == 2]
    #s1_data = data[data['type'] == 1]
    print(label_vec_nonzero)
    print(len(label_vec_nonzero))
    print(type(label_vec_nonzero))
    data_new['type'] = label_vec_nonzero.astype(int)
    
    # match time index to original time
    # hate with a passion that I have to do it with a for loop but it is what it is
    # for samples in np.arange(len(data_new)):
    #    np.ar
    
    return colorp, data_new

def affine_transform(data: np.ndarray, data_min: float, data_max: float, target_min: float, target_max: float) -> np.ndarray:
    """
    Takes a set of data an applies a affine transfrom to scale it
    data: data
    data_min = minimum of your data set
    data_max = maximum of the data set
    target_min = minimum of the target space
    target_max = maximum of the target space
    """
    D = ((data - data_min)/(data_max-data_min))*(target_max-target_min) + target_min
    return D
    
# I should make a separate file for image manipulation functions
def select_middle_pixel(img_as_np_array: np.ndarray, pxl_per_block: int = 12) -> np.ndarray:
    """
    Image resulting from NS have cells of about 12 pixels, we want to reduce the
    image to 1 pixel per cell, so we will take the middle pixel.
    Since images have their 0 index at the top and np arrays start at the bottom
    we have to filp the image across the y-axis.
    """
    [width, height, depth] = img_as_np_array.shape
    #img_flipped = np.flip(img_as_np_array, 0) # image indexing start at the top and go down
                                              # this fixes this issue.
    img_flipped = img_as_np_array
    SOM_width = int(width/12)
    SOM_height = int(height/12)
    
    SOM_img_clusters = np.zeros([SOM_width, SOM_height, depth])
    
    for col in np.arange(SOM_width):
        #print(f'col number is : {col}')
        for row in np.arange(SOM_height):
            #print(f'Number in computation is {pxl_per_block/2 + (row*12)}')
            SOM_img_clusters[col, row, :] = img_flipped[int(pxl_per_block/2) + (col*12), 
                                                        int(pxl_per_block/2) + (row*12), :]
            
    return SOM_img_clusters

def recall_populations(dataset: np.ndarray, weight_cube: np.ndarray, SOM_cls_img: np.ndarray, norm_factors: np.ndarray) -> np.ndarray:
    """
    Master function that should let the user provide a weightcube,
    a reference img as a np.array, a dataset and a set of normalization factors.
    In theory, if these 5 things are provided, this function should output
    the original data back with one added field with the name "SOM_type"
    Here we will assume that the data has been preprocessed in the SOM
    input format.

    weight_cube:      SOM weight cube (3D array)
    SOM_cls_img:      SOM reference image as a numpy array
    dataset:          Data to preform the recall on should be a structured array
    normfactos:       A set of 11 numbers to normalize the data so we can preform a recall
    """
    [SOM_xdim, SOM_ydim, SOM_zdim] = weight_cube.shape
    [IMG_xdim, IMG_ydim, IMG_zdim] = SOM_cls_img.shape
    unique_colors = np.unique(np.reshape(SOM_cls_img, [SOM_xdim * SOM_ydim, 3]), axis=0)
    # Checks that the reference image matches the weight cube
    assert SOM_xdim == IMG_xdim, f'Dimensions mismatch between SOM weight cube ({SOM_xdim}) and reference image ({IMG_xdim})'
    assert SOM_ydim == IMG_ydim, f'Dimensions mismatch between SOM weight cube ({SOM_ydim}) and reference image ({IMG_ydim})'

    # Get the deciles representation of data for recall
    #decile_transform_check = data_to_log_decile_log_area_aft(dataset, norm_factors)
    # preform a recall of the dataset with the weight cube
    # assign each population color a number (can do from previous function)
    ref_map = generate_color_ref_map(SOM_cls_img, unique_colors, SOM_xdim, SOM_ydim)
    SOM_cls_array = np.empty(len(dataset))
    SOM_cls_array[:] = np.nan
    # Make new numpy structured array to save the SOM cls data
    data_with_SOM_cls = rfn.append_fields(dataset, 'SOM_type', SOM_cls_array)
    # preforms the recall and assigns SOM_type label
    output_data = SOM_cls_recall(data_with_SOM_cls, dataset, weight_cube, ref_map)
    return output_data['SOM_type']


def generate_color_ref_map(color_image: np.ndarray, unique_colors: np.ndarray) -> np.ndarray:
    """
    Generate a map where the color image representing the labels of the
    som weight cube.

    ==================================
    Inputs
    color_image:   image made by the remap compressed to the SOM size
    unique_colors: unique colors found in the image (also represent # of clusters)
    """
    xdim, ydim, _ = np.shape((color_image))
    ref_map = np.zeros((xdim, ydim))
    for color in np.arange(len(unique_colors)):
        mask = np.all(np.equal(color_image, unique_colors[color, :]), axis=2)
        indices = np.argwhere(mask)  # generates a 2d mask
        for loc in np.arange(len(indices)):
            ref_map[indices[loc][0], indices[loc][1]] = color
    return ref_map


def SOM_cls_recall(array_to_fill: np.ndarray, data_in_SOM_fmt: np.ndarray, weight_cube: np.ndarray, reference_map: np.ndarray) -> np.ndarray:
    """
    Takes the data, the weight cube and the classification map and assignes each
    data point a label based on their cluster.

    array_to_fill:    empty array for the som labels
    data_in_SOM_fmt:  the peak data in the form of the input to the SOM
    weight_cube:      some weight cube
    reference_map:    color map with som classification
    """

    # Want to make it so it works with different metrics in the future
    [SOM_xdim, SOM_ydim, _] = weight_cube.shape
    distances = cdist(weight_cube.reshape(-1, weight_cube.shape[-1]), data_in_SOM_fmt, metric='euclidean')
    w_neuron = np.argmin(distances, axis=0)
    x_idx, y_idx = np.unravel_index(w_neuron, (SOM_xdim, SOM_ydim))
    array_to_fill['SOM_type'] = reference_map[x_idx, y_idx]
    return array_to_fill

def create_mapping_dict(output_classes: np.ndarray, dataset_classes: np.ndarray) -> Dict:
    """
    Create a mapping dictionary from output classes to dataset classes.

    Parameters:
    output_classes (list): List of output classes from the neural network.
    dataset_classes (list): List of corresponding dataset classes.

    Returns:
    dict: A dictionary mapping output classes to dataset classes.
    """
    if len(output_classes) != len(dataset_classes):
        raise ValueError("The number of output classes must match the number of dataset classes.")
    return dict(zip(output_classes, dataset_classes))

def map_output_to_dataset(output_classes: np.ndarray, mapping_dict: np.ndarray) -> np.ndarray:

    """
    Map output classes to dataset classes using the mapping dictionary.

    Parameters:
    output_array (np.ndarray): Array of output classes from the neural network.
    mapping_dict (dict): Dictionary mapping output classes to dataset classes.

    Returns:
    np.ndarray: Array of dataset classes corresponding to the output classes.
    """
    mapped_array = np.vectorize(mapping_dict.get)(output_classes)
    return mapped_array
            
    
def normalize_data_recall(peaklet_data, normalization_factor):
    """
    Use this function to do operation with an already trained SOM
    Converts peaklet data into the current best inputs for the SOM,
    log10(deciles) + log10(area) + AFT
    Since we are dealing with logs, anything less than 1 will be set to 1

    peaklet_data:           straxen datatype peaklets (peaks also work)
    normalization_factors:  numbers needed to normalize data so recalls work
    """
    pass

    


