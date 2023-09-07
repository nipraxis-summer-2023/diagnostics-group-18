""" Scan outlier metrics
"""

# Any imports you need
# +++your code here+++
import numpy as np

def dvars(img):
    """ Calculate dvars metric on Nibabel image `img`

    The dvars calculation between two volumes is defined as the square root of
    (the mean of the (voxel differences squared)).

    Parameters
    ----------
    img : nibabel image

    Returns
    -------
    dvals : 1D array
        One-dimensional array with n-1 elements, where n is the number of
        volumes in `img`.
    """

    data = img.get_fdata()
    # create two timeseries with n-1 volumes and find the difference between them
    # remove the last volume
    img_start = data[...,:-1]
    # remove the first volume
    img_end = data[..., 1:]

    return np.sqrt(np.mean((img_start - img_end)**2, axis = (0,1,2)))
    raise NotImplementedError('Code up this function')
