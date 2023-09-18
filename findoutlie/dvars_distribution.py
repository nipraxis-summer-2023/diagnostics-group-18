"""
Functions to calculate variables for a standard DVARS distribution given a 
timeseries of volumes

For more information and proofs see: 
https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE


"""

import numpy as np

def distribution_mean(data):
    """ Calculate mean of standard distribution on Nibabel image `img`
    used to calculate p-value for DVARS metrics

    Calculate the sum of variances for voxel intensities 
    and divide by the sum of intensities

    For more information see equation 12: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    img : nibabel image

    Returns
    -------
    mu_0 : float
        decimal to describe to mean for the standard distribution of the timeseries
    """
    # calculate the variance of all voxel timeseries
    voxel_variances = np.var(data, axis=3)
    # sum of variances divided by sum of all intensities
    return np.sum(voxel_variances)/np.sum(data)