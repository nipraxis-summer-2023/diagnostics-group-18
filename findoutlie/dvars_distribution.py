"""
Functions to calculate variables for a standard DVARS distribution given a 
timeseries of volumes

For more information and proofs see: 
https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE


"""

import numpy as np
from scipy.special import ndtri
import scipy.stats

def dvars_pvals(raw_data, alpha=0.05):
    """ Calculate Bonferroni corrected p_values using chi2 stats and finds 
    rejected null hypothesys

    Parameters
    ----------
    raw_data: 4D timeseries
    alpha: significance threshold

    Returns
    -------
    p_vals_adjusted: 1D array of adjusted p_values
    reject_null: 1D array, true if null hypothesys is rejected 
    """
    # scale the data
    data = volume_scaling(raw_data)
    
    # calculate chi2 stats
    chisquared_stats, degrees_of_freedom = dvars_chisquared_stat(data)
    
    # find p-vals
    p_vals = 1 - scipy.stats.chi2.cdf(chisquared_stats,degrees_of_freedom)
    
    # Bonferroni adjusted p_values
    p_vals = p_vals*len(p_vals)
    
    # significance testing
    reject_null = p_vals <= alpha
    
    return p_vals, reject_null

def volume_scaling(raw_data): 
    """ Rescales volume intensity values from raw
    4D timeseries
    
    Maths:
    
    M_Ri = raw mean value for voxel i
    Y_Rit = raw value for voxel i in volume t
    m_R = mean (or median) raw value of {M_Ri}

    Scaled value for voxel i at time t
    Y_it = 100*(Y_Rit-M_Ri)/m_R 

    see equation 1: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    raw_data: raw 4D timeseries 

    Returns
    -------
    data : Scaled 4D timeseries
    """

    # overall mean value 
    m_R = np.mean(raw_data)

    scaled_data = np.zeros(raw_data.shape)
    
    # initiate loop over x,y,z 
    for x_value in range(raw_data.shape[0]):
        for y_value in range(raw_data.shape[1]):
            for z_value in range(raw_data.shape[2]):
                    
                    # calculate mean for voxel timeseries
                    M_Ri = np.mean(raw_data[x_value,y_value,z_value,:])
                    
                    # loop over volumes
                    for t_value in range(raw_data.shape[3]):
                        
                        # get individual raw intensities
                        Y_Rit = raw_data[x_value,y_value,z_value,t_value]
                        # calculate scaled intensity
                        Y_it = 100*(Y_Rit-M_Ri)/m_R 
                        scaled_data[x_value,y_value,z_value,t_value] = Y_it
    
    return scaled_data

def dvars_chisquared_stat(data):
    """ Calculate chi squared stats for and degrees of freedom for 4D timeseries 

    Maths:

    X(dvars_t) = 2*mu_0(dvars^2_t)/(variance_0)

    dof = 2*mu_0^2/(variance_0)

    For more information see equation 11: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    data: 4D timeseries

    Returns
    -------
    chi_squared_stats: 1D array
        One-dimensional array with n-1 elements, where n is the number of
        volumes in data.
    dof: degrees of freedom
    """
    # calculate dvars^2 values
    dvars_squared_values = dvars_squared(data)
    
    # calculate mean for the null distribution
    null_mean = null_distribution_mean(data)
    #null_mean = np.median(dvars_squared_values)

    # calculate the variance for the null distribution
    null_variance = null_distribution_variance(data)

    # calculate chi squared stats
    return 2*null_mean*(dvars_squared_values)/(null_variance), 2*null_mean**2/(null_variance)

def dvars_squared(data):
    """ Calculate dvars^2 from 4D data

    The dvars calculation between two volumes is defined as the square root of
    (the mean of the (voxel differences squared)).

    Parameters
    ----------
    data : 4D volumes

    Returns
    -------
    dvals^2 : 1D array
        One-dimensional array with n-1 elements, where n is the number of
        volumes in data.
    """

    return np.mean((voxel_differences(data))**2, axis = (0,1,2))

def voxel_differences(data):
    """ Calculate difference between voxel i at time t 
    and the same voxel at time t + 1 
    
    Maths:

    Y_i(t+1)-Y_i(t)

    Parameters
    ----------
    data: 4D timeseries of shape (X,Y,Z,T)

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

def null_distribution_mean(data):
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
        estimate of the mean for the null distribution of the timeseries
    """
    # calculate the IQR of all voxel timeseries
    voxel_iqrs = voxel_iqr_variance(data)

    # the median of the iqr 
    return np.median(voxel_iqrs)

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
    # all_voxel_differences = np.sqrt(dvars_squared(data))

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

def null_distribution_variance(data):
    """ Calculate estimate of variance for null distribution on 4D data

    Maths:

    Step 1
    Calculate dvars_squared (from function)

    Step 2
    Find hIQR({dvars^2_t}) the half IQR defined as the difference between
    the median and lower quartile.

    Step 3
    Calculate the variance of the null distribution = hIQR({dvars^2_t})/hIQR_0 
    where hIQR_0 is half of the IQR for the standard normal distribution

    For more information see equation 14: 
    https://www.sciencedirect.com/science/article/pii/S1053811917311229?via%3Dihub#appsecE

    Parameters
    ----------
    data: 4D timeseries

    Returns
    -------
    variance_0 : float
        estimate of the variance for the null distribution of the timeseries
    """

    # calculate dvars values
    dvars_squared_values = dvars_squared(data)

    # calculate hIQR for dvars values
    q1, q2 = np.percentile(dvars_squared_values, [25, 50])
    hIQR = q2 - q1

    # define hIQR_0 and calculate null distribution variance
    hIQR_0 = (ndtri(0.75) - ndtri(0.25))/2
    return hIQR/hIQR_0
