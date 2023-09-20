""" Scan outlier metrics
"""

import numpy as np
import nibabel as nib

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
    # validation if it is nifty image
    if not isinstance(img, nib.Nifti1Image):
        raise TypeError("Input must be a nibabel image object")

    # loading data into a memory
    data = img.get_fdata()

    # validation if it 4D images
    if len(data.shape) != 4:
        raise ValueError("Input image must be 4D")

    # remove the last volume
    img_start = data[...,:-1]

    # remove the first volume
    img_end = data[..., 1:]

    return np.sqrt(np.mean((img_start - img_end) ** 2, axis = (0,1,2)))
    # raise NotImplementedError('Code up this function')
