""" Module with routines for finding outliers
"""

from pathlib import Path
from .metrics import dvars
from .detectors import iqr_detector
import nibabel as nib
import numpy as np


def detect_outliers(fname):
    img = nib.load(fname)
    data = img.get_fdata()
    mean_values = np.mean(data, axis=(0, 1, 2))
    outliers = np.where(iqr_detector(mean_values))[0]
    return outliers
    # img = nib.load(fname)
    # # img_data = img.get_fdata()
    # dvars_values = dvars(img)
    # # print(f"Min DVARS: {min(dvars_values)}, Max DVARS: {max(dvars_values)}")
    # # print(f"First 10 DVARS values: {dvars_values[:10]}")
    # some_threshold = 40
    # outliers = [i for i, val in enumerate(dvars_values) if val > some_threshold]
    return outliers


def find_outliers(data_directory):
    """ Return filenames and outlier indices for images in `data_directory`.

    Parameters
    ----------
    data_directory : str
        Directory containing containing images.

    Returns
    -------
    outlier_dict : dict
        Dictionary with keys being filenames and values being lists of outliers
        for filename.
    """
    image_fnames = Path(data_directory).glob('**/sub-*.nii.gz')
    outlier_dict = {}
    for fname in image_fnames:
        outliers = detect_outliers(fname)
        outlier_dict[fname] = outliers
    return outlier_dict
