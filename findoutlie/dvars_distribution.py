"""
Functions to calculate variables for a standard DVARS distribution given a 
timeseries of volumes

For more information and proofs see: 
https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE


"""

import numpy as np
import scipy
from scipy.special import ndtri


def distribution_mean(data):
    """ Calculate mean of null distribution on 4D data
    using robust IQR measurements of voxel intensity differences.

    Maths:

    Step 1
    Calculate IQR_i/IQR_0 where IQR_i is the IQR for voxel timeseries
    in location i and IQR_0 is the IQR for the standard null distribution.
    see function voxel_iqr_variance

    Step 2
    The median of {IQR_i/IQR_0} is an estimate for the mean of the null
    distribution, other estimates are available

    For more information see equation 12 and 13: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    data: 4D timeseries

    Returns
    -------
    mu_0 : float
        decimal to describe the mean for the null distribution of the timeseries
    """
    # calculate the variance of all voxel timeseries
    voxel_variances = voxel_iqr_variance(data)

    # the median of the iqr 
    return np.median(voxel_variances)

def voxel_differences(data):
    """ Calculate difference between voxel i at time t 
    and the same voxel at time t + 1 
    
    Maths:

    V_i(t+1)-V_i(t)

    Parameters
    ----------
    data: normalised 4D timeseries of shape (X,Y,Z,T)

    Returns
    -------
    dvals : 4D array with one less volume than 'data' of shape (X,Y,Z,T-1)
    """
    # create two timeseries with n-1 volumes and find the difference between them
    # remove the last volume
    img_start = data[...,:-1]
    # remove the first volume
    img_end = data[..., 1:]
    return img_start - img_end

def voxel_iqr_variance(data):
    """ Calculate IQR of voxel timeseries differences
    used to calculate mean of null distribution

    For more information see equation 13: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    data: 4D timeseries

    Returns
    -------
    IQR_values: 3D array of IQR for each voxel timeseries difference
    """
    # from the data calculate the differences between voxels at t and t+1
    all_voxel_differences = voxel_differences(data)
    
    # initiate variable for loop
    iqr_dvars_values = np.zeros(data.shape[:-1])

    # initiate loop over x,y,z 
    for x_value in range(data.shape[0]):
        for y_value in range(data.shape[1]):
            for z_value in range(data.shape[2]):
                
                # calculate iqr for each voxel timeseries
                q1, q3 = np.percentile(all_voxel_differences[x_value,y_value,z_value,:], [25, 75])
                iqr_dvars_values[x_value,y_value,z_value] = q3 - q1

    # IQR of standard normal distribution
    iqr_0 = ndtri(0.75) - ndtri(0.25)

    return iqr_dvars_values/iqr_0