import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy.lib.recfunctions as rfn
import strax
import straxen
import viff

def assign_labels(data, ref_img, xdim, ydim, cut_out):
    '''This functions takes in the data and classifications based on an image gives the
    unique labels as well as the data set bacl with the new classification
    PS this version only takes in S1s and S2s and ignores unclassified samples, 
    another version will be made to deal with the unclassified samples
    
    data: can be either peaks or peak_basics
    ref_img: will be the image extracted from the SOM classification of each data point
    xdim: width of the image cube
    ydim: height of the image cube
    cut_out: removes the n last digits of the image vector if necesarry'''
    from PIL import Image
    data_new = data
    img = Image.open(ref_img)
    imgGray = img.convert('L')
    #imgGray2.save('/home/luissanchez25/im_kr83_real__30x30_2lbl.0.rmpmap.png')
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
    #for samples in np.arange(len(data_new)):
    #    np.ar
    
    return colorp, data_new

def affine_transform(data, data_min, data_max, target_min, target_max):
    """
    takes a set of data an applies a affine transfrom to scale it
    data: data
    data_min = minimum of your data set
    data_max = maximum of the data set
    target_min = minimum of the target space
    target_max = maximum of the target space
    """
    D = ((data - data_min)/(data_max-data_min))*(target_max-target_min) + target_min
    return D

def import_khoros_weightcube(path_to_weights):
    """
    Imports a weightcube generated with the khoros system,
    reshapes it into the appropriate format and applies an
    affine transform for recalls

    path_to_weights: path to where the som weight is located
    """
    import viff
    
    wgtcub = viff.read(path_to_weights)
    [_, zdim, xdim, ydim] = wgtcub.shape
    wgtcub_re = np.reshape(wgtcub, [zdim, xdim, ydim])
    wgtcub_tr = np.transpose(wgtcub_re, [1,2,0])
    weight_cube = affine_transform(wgtcub_tr, -1,1,0,1)
    return weight_cube
    
def select_middle_pixel(img_as_np_array, pxl_per_block = 12):
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

def recall_populations(dataset, weight_cube, SOM_cls_img, norm_factors):
    """
    Master function that should let the user provide a weightcube,
    a reference img as a np.array, a dataset and a set of normalization factors.
    In theory, if these 5 things are provided, this function should output
    the original data back with one added field with the name "SOM_type"
    weight_cube:      SOM weight cube (3D array)
    SOM_cls_img:      SOM reference image as a numpy array
    dataset:          Data to preform the recall on (Should be peaklet level data)
    normfactos:       A set of 11 numbers to normalize the data so we can preform a recall
    """
    [SOM_xdim, SOM_ydim, SOM_zdim] = weight_cube.shape
    [IMG_xdim, IMG_ydim, IMG_zdim] = SOM_cls_img.shape
    unique_colors = np.unique(np.reshape(SOM_cls_img, [SOM_xdim * SOM_ydim, 3]), axis=0)
    # Checks that the reference image matches the weight cube
    assert SOM_xdim == IMG_xdim, f'Dimensions mismatch between SOM weight cube ({SOM_xdim}) and reference image ({IMG_xdim})'
    assert SOM_ydim == IMG_ydim, f'Dimensions mismatch between SOM weight cube ({SOM_ydim}) and reference image ({IMG_ydim})'

    assert all(dataset['type'] != 0), 'Dataset contains unclassified peaklets'
    # Get the deciles representation of data for recall
    decile_transform_check = data_to_log_decile_log_area_aft(dataset, norm_factors)
    # preform a recall of the dataset with the weight cube
    # assign each population color a number (can do from previous function)
    ref_map = generate_color_ref_map(SOM_cls_img, unique_colors, SOM_xdim, SOM_ydim)
    SOM_cls_array = np.empty(len(dataset['area']))
    SOM_cls_array[:] = np.nan
    # Make new numpy structured array to save the SOM cls data
    data_with_SOM_cls = rfn.append_fields(dataset, 'SOM_type', SOM_cls_array)
    # preforms the recall and assigns SOM_type label
    output_data = SOM_cls_recall(data_with_SOM_cls, decile_transform_check, weight_cube, ref_map)
    return output_data['SOM_type']


def generate_color_ref_map(color_image, unique_colors):
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


def SOM_cls_recall(array_to_fill, data_in_SOM_fmt, weight_cube, reference_map):
    """
    Takes the data, the weight cube and the classification map and assignes each
    data point a label based on their cluster.

    array_to_fill:    empty array for the som labels
    data_in_SOM_fmt:  the peak data in the form of the input to the SOM
    weight_cube:      some weight cube
    reference_map:    color map with som classification
    """
    [SOM_xdim, SOM_ydim, _] = weight_cube.shape
    distances = cdist(weight_cube.reshape(-1, weight_cube.shape[-1]), data_in_SOM_fmt, metric='euclidean')
    w_neuron = np.argmin(distances, axis=0)
    x_idx, y_idx = np.unravel_index(w_neuron, (SOM_xdim, SOM_ydim))
    array_to_fill['SOM_type'] = reference_map[x_idx, y_idx]
    return array_to_fill


def som_type_to_type(som_type, s1_array, s2_array):
    """
    Converts the SOM type into either S1 or S2 type (1, 2)
    som_type:    array with integers corresponding to the different SOM types
    s1_array:    array containing the number corresponding to the SOM types which should
                 be converted to S1's
    """
    som_type_copy = som_type.copy()
    som_type_copy[np.isin(som_type_copy, s1_array)] = 1234
    som_type_copy[np.isin(som_type_copy, s2_array)] = 5678
    som_type_copy[som_type_copy == 1234] = 1
    som_type_copy[som_type_copy == 5678] = 2
    assert np.all(np.unique(som_type_copy) == np.array([1, 2])), f'Error, values other than s1 and s2 found in the array'
    return som_type_copy
            
def save_khoros_raw(file_name, data):
    """
    Saves a given data into the desired raw data file so I can use 
    it to train an SOM in NeuroScope.
    
    file_name:    Path to where you want to save the file + file name
    data:         Data for the weightcube to use in neuroscope
    
    """
    import viff
    
    [Length, Width, Height] = np.shape(data)
    save_khoros_raw = np.reshape(data.transpose(2,1,0), [1, Height, Length, Width])
    save_khoros_raw_c = np.ascontiguousarray(save_khoros_raw)
    assert '.raw' in file_name, "The output file must be a raw file!"
    viff.write(file_name, save_khoros_raw_c)
    
    
def data_to_log_decile_log_area_aft_recall(peaklet_data, normalization_factor):
    """
    Use this function to do operation with an already trained SOM
    Converts peaklet data into the current best inputs for the SOM,
    log10(deciles) + log10(area) + AFT
    Since we are dealing with logs, anything less than 1 will be set to 1

    peaklet_data:           straxen datatype peaklets (peaks also work)
    normalization_factors:  numbers needed to normalize data so recalls work
    """
    # turn deciles into approriate 'normalized' format (maybe also consider L1 normalization of these inputs)
    decile_data = compute_quantiles(peaklet_data, 10)
    data = peaklet_data.copy()
    decile_data[decile_data < 1] = 1
    # decile_L1 = np.log10(decile_data)
    decile_log = np.log10(decile_data)
    decile_log_over_max = np.divide(decile_log, normalization_factor[:10])
    # Now lets deal with area
    data['area'] = data['area'] + normalization_factor[11] + 1
    peaklet_log_area = np.log10(data['area'])
    peaklet_aft = np.sum(data['area_per_channel'][:, :straxen.n_top_pmts], axis=1) / peaklet_data['area']
    peaklet_aft = np.where(peaklet_aft > 0, peaklet_aft, 0)
    peaklet_aft = np.where(peaklet_aft < 1, peaklet_aft, 1)
    deciles_area_aft = np.concatenate((decile_log_over_max,
                                       np.reshape(peaklet_log_area, (len(peaklet_log_area), 1)) / normalization_factor[
                                           10],
                                       np.reshape(peaklet_aft, (len(peaklet_log_area), 1))), axis=1)
    return deciles_area_aft

    
def data_to_log_decile_log_area_aft_generate(peaklet_data):
    """
    Use this function for generating data to train an SOM
    Converts peaklet data into the current best inputs for the SOM,
    log10(deciles) + log10(area) + AFT
    Since we are dealing with logs, anything less than 1 will be set to 1
    Lets explain the norm factors:
    0->9 max of the log of each decile => normalizes it to 1
    10 -> max of the log of the area => to normalize to 1
    11 -> keep track of the minimum value used to add to all other data
    """
    # turn deciles into approriate 'normalized' format (maybe also consider L1 normalization of these inputs)
    decile_data = compute_quantiles(peaklet_data, 10)
    data = peaklet_data.copy()
    decile_data[decile_data < 1] = 1
    # decile_L1 = np.log10(decile_data)
    decile_log = np.log10(decile_data)
    log_max = np.max(decile_log, axis = 0)
    print(log_max)
    decile_log_over_max = np.divide(decile_log, log_max)
    # Now lets deal with area
    min_area = np.min(data['area'])
    print(min_area)
    data['area'] = data['area'] + np.abs(min_area) + 1
    peaklet_log_area = np.log10(data['area'])
    peaklet_aft = np.sum(data['area_per_channel'][:, :straxen.n_top_pmts], axis=1) / peaklet_data['area']
    peaklet_aft = np.where(peaklet_aft > 0, peaklet_aft, 0)
    peaklet_aft = np.where(peaklet_aft < 1, peaklet_aft, 1)
    deciles_area_aft = np.concatenate((decile_log_over_max,
                                       np.reshape(peaklet_log_area, (len(peaklet_log_area), 1)) / np.max(peaklet_log_area),
                                       np.reshape(peaklet_aft, (len(peaklet_log_area), 1))), axis=1)
    
    norm_factors = np.concatenate((np.max(decile_log, axis = 0), np.max(np.log10(peaklet_data['area'])).reshape(1)))
    norm_factors = np.concatenate((norm_factors, np.abs(np.min(peaklet_data['area'])).reshape(1)))
    
    return deciles_area_aft, norm_factors


def compute_quantiles(peaks: np.ndarray, n_samples: int):
    """
    Compute waveforms and quantiles for a given number of nodes(attributes)
    :param peaks:
    :param n_samples: number of nodes or attributes
    :return:quantiles
    """
    data = peaks['data'].copy()
    data[data < 0.0] = 0.0
    dt = peaks['dt']
    q, wf = strax.compute_wf_attributes(data, dt, n_samples, False)
    return q
