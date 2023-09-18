"""
Functions to calculate variables for a standard DVARS distribution given a 
timeseries of volumes

For more information and proofs see: 
https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE


"""

import numpy as np

def distribution_mean(data):
    """ Calculate mean of standard distribution on 4D data
    used to calculate p-value for DVARS metrics

    Calculate the sum of variances for voxel intensities 
    and divide by the sum of intensities

    For more information see equation 12: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    data: 4D timeseries

    Returns
    -------
    mu_0 : float
        decimal to describe to mean for the standard distribution of the timeseries
    """
    # calculate the variance of all voxel timeseries
    voxel_variances = np.var(data, axis=3)
    # sum of variances divided by sum of all intensities
    return np.sum(voxel_variances)/np.sum(data)

def dvars_data(data):
    """ Calculate dvars metric on 4D data

    The dvars calculation between two volumes is defined as the square root of
    (the mean of the (voxel differences squared)).

    Parameters
    ----------
    data: 4D timeseries

    Returns
    -------
    dvals : 1D array
        One-dimensional array with n-1 elements, where n is the number of
        volumes in data.
    """
    # create two timeseries with n-1 volumes and find the difference between them
    # remove the last volume
    img_start = data[...,:-1]
    # remove the first volume
    img_end = data[..., 1:]
    return np.sqrt(np.mean((img_start - img_end)**2, axis = (0,1,2)))

def distribution_variance(data):

    dvars_data(data):