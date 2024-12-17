import numba
import numpy as np
try:
    import straxen
    HAS_STRAXEN = True
except ImportError:
    HAS_STRAXEN = False

def data_to_log_decile_log_area_aft(peaklet_data: np.ndarray, 
                                    normalization_factor: np.ndarray) -> np.ndarray:
    """
    Takes peakelt level data and converts it into input vectors for the SOM consisting of: deciles, log10(area), AFT
    
    Converts peaklet data into the current best inputs for the SOM, log10(deciles) + log10(area)
    + AFT Since we are dealing with logs, anything less than 1 will be set to 1. 
    Any decile value < 1 will be set to 1 before log10 is taken.
    The areas can be very negative to we add the minimum value to all areas to make them positive.

    Parameters
    ----------
    peaklet_data : np.ndarray
        Peaklet level data
    normalization_factor : np.ndarray
        Normalization factors for the data
    
    Returns
    -------
    deciles_area_aft : np.ndarray
        Normalized Input vectors for the SOM
    
    """
    # turn deciles into approriate 'normalized' format
    # (maybe also consider L1 normalization of these inputs)

    if not HAS_STRAXEN:
        raise ImportError("straxen is not installed. Please install straxen to use this function")
    decile_data = compute_quantiles(peaklet_data, 10)
    data = peaklet_data.copy()
    decile_data[decile_data < 1] = 1

    decile_log = np.log10(decile_data)
    decile_log_over_max = np.divide(decile_log, normalization_factor[:10])
    # Now lets deal with area
    data["area"] = data["area"] + normalization_factor[11] + 1
    peaklet_log_area = np.log10(data["area"])
    peaklet_aft = (
        np.sum(data["area_per_channel"][:, : straxen.n_top_pmts], axis=1) / peaklet_data["area"]
    )
    peaklet_aft = np.where(peaklet_aft > 0, peaklet_aft, 0)
    peaklet_aft = np.where(peaklet_aft < 1, peaklet_aft, 1)
    deciles_area_aft = np.concatenate(
        (
            decile_log_over_max,
            np.reshape(peaklet_log_area, (len(peaklet_log_area), 1)) / normalization_factor[10],
            np.reshape(peaklet_aft, (len(peaklet_log_area), 1)),
        ),
        axis=1,
    )
    return deciles_area_aft

def compute_quantiles(peaks: np.ndarray, n_samples: int):
    """
    Compute waveforms and quantiles for a given number of nodes(attributes)

    Parameters
    ----------
    peaks : np.ndarray
        Peaks data
    n_samples : int
        Number of nodes or attributes

    Returns
    -------
    quantiles : np.ndarray
        Quantiles of the waveform

    """

    data = peaks["data"].copy()
    data[data < 0.0] = 0.0
    dt = peaks["dt"]
    q = compute_wf_attributes(data, dt, n_samples)
    return q


#@export
@numba.jit(nopython=True, cache=True)
def compute_wf_attributes(data, sample_length, n_samples: int):
    """
    Compute waveform attribures.

    Quantiles: represent the amount of time elapsed for
    a given fraction of the total waveform area to be observed in n_samples
    i.e. n_samples = 10, then quantiles are equivalent deciles.

    Parameters
    ----------
    data : np.ndarray
        Waveform data
    sample_length : np.ndarray
        Length of each sample
    n_samples : int
        Number of samples

    Returns
    -------
    quantiles : np.ndarray
        Quantiles of the waveform
    """

    assert data.shape[0] == len(sample_length), "ararys must have same size"

    num_samples = data.shape[1]

    quantiles = np.zeros((len(data), n_samples), dtype=np.float64)

    # Cannot compute with with more samples than actual waveform sample
    assert num_samples > n_samples, "cannot compute with more samples than the actual waveform"
    assert num_samples % n_samples == 0, "number of samples must be a multiple of n_samples"

    # Compute quantiles
    inter_points = np.linspace(0.0, 1.0 - (1.0 / n_samples), n_samples)
    cumsum_steps = np.zeros(n_samples + 1, dtype=np.float64)
    frac_of_cumsum = np.zeros(num_samples + 1)
    sample_number_div_dt = np.arange(0, num_samples + 1, 1)
    for i, (samples, dt) in enumerate(zip(data, sample_length)):
        if np.sum(samples) == 0:
            continue
        # reset buffers
        frac_of_cumsum[:] = 0
        cumsum_steps[:] = 0
        frac_of_cumsum[1:] = np.cumsum(samples)
        frac_of_cumsum[1:] = frac_of_cumsum[1:] / frac_of_cumsum[-1]
        cumsum_steps[:-1] = np.interp(inter_points, frac_of_cumsum, sample_number_div_dt * dt)
        cumsum_steps[-1] = sample_number_div_dt[-1] * dt
        quantiles[i] = cumsum_steps[1:] - cumsum_steps[:-1]

    return quantiles


### Functions beyond this point will not be used
# Keeping this here just in case

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
    if not HAS_STRAXEN:
        raise ImportError("straxen is not installed. Please install straxen to use this function")

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




