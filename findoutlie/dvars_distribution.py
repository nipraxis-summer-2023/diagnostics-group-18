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
    used to calculate p-value for DVARS metrics

    Calculate the mean variances of voxel intensities 


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

    # sum of variances divided by voxels per volume (mean)
    return np.mean(voxel_variances)

def voxel_differences(data):
    """ Calculate difference between voxel i at time t 
    and the same voxel at time t + 1 
    
    V(t+1)-V(t)

    Parameters
    ----------
    data: normalised 4D timeseries

    Returns
    -------
    dvals : 4D array with one less volume than 'data'
    """
    # create two timeseries with n-1 volumes and find the difference between them
    # remove the last volume
    img_start = data[...,:-1]
    # remove the first volume
    img_end = data[..., 1:]
    return img_start - img_end

def voxel_iqr_variance(data):
    """ Calculate variance of voxel timeseries using iqr
    used to calculate mean of null distribution

    For more information see equation 13: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    data: 4D difference of adjacent voxel intensities

    Returns
    -------
    variance: 3D array of variances for each voxel
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

    # IQR os standard normal distribution
    iqr_0 = ndtri(0.75) - ndtri(0.25)

    return iqr_dvars_values/iqr_0