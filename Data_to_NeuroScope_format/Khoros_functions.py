import viff

def import_khoros_weightcube(path_to_weights):
    """
    Imports a weightcube generated with the khoros system,
    reshapes it into the appropriate format and applies an
    affine transform for recalls
    
    """
    
    wgtcub = viff.read(path_to_weights)
    [_, zdim, xdim, ydim] = wgtcub.shape
    wgtcub_re = np.reshape(wgtcub, [zdim, xdim, ydim])
    wgtcub_tr = np.transpose(wgtcub_re, [1,2,0])
    weight_cube = affine_transform(wgtcub_tr, -1,1,0,1)
    return weight_cube

def data_to_raw_file_4_khoros(data, file: str):
    """
    Make data file into an appropriate raw file for khoros format
    
    data:    3D data cube
    file:    path location + filename of desired output file
    """
    data_t = np.transpose(data, [2,0,1])
    np.asfortranarray(data_t.astype('float64')).tofile(file)
    print('Data has been saved')