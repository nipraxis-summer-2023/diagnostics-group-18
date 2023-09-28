""" Test script for detector functions

Run these tests with::

    python3 findoutlie/tests/test_dvars_distribution.py

or better, in IPython::

    %run findoutlie/tests/test_dvars_distribution.py

or even better, from the terminal::

    pytest findoutlie/tests/test_dvars_distribution.py

"""

import numpy as np
from findoutlie.dvars_distribution import null_distribution_mean, voxel_differences, voxel_iqr_variance, dvars_squared
from scipy.special import ndtri
from math import isclose
import nipraxis as npx
import nibabel as nib

# create test volume timeseries filled with normally distributed random numbers
TEST_NORMALISED_VOLUMES = np.random.normal(size = (10,10,10,1000))


def test_null_distribution_mean():
    
    # create example matrices 
    example_values1 = np.ones((3,3,3,6))
    example_values2 = np.zeros((3,3,3,12))

    for time_values in range(example_values2.shape[-1]):
        example_values2[...,time_values] = time_values**2
    
    example_answer2_array = voxel_differences(example_values2)[1,1,1,:]
    q1, q3 = np.percentile(example_answer2_array, [25, 75])
    answer2 = (q3 - q1)/(ndtri(0.75) - ndtri(0.25))

    assert null_distribution_mean(example_values1) == 0
    assert isclose(null_distribution_mean(example_values2), answer2)
    #assert isclose(null_distribution_mean(TEST_NORMALISED_VOLUMES), 0)


def test_voxel_differences():
    example_values = np.ones((3,3,3,6))
    example_values1 = np.append(example_values,example_values+1, axis=3)
    example_answer3 = TEST_NORMALISED_VOLUMES[...,1]-TEST_NORMALISED_VOLUMES[...,0]

    # check correct shape for answer
    assert voxel_differences(example_values).shape == (3,3,3,5)

    # check correct order of operations and values
    example_answer1 = np.zeros(example_values.shape[:-1]+(example_values.shape[-1] - 1,))
    example_answer1 = np.append(example_answer1,example_values, axis=3)
    assert (voxel_differences(example_values1) == example_answer1).all
    assert (voxel_differences(TEST_NORMALISED_VOLUMES)[...,0] == example_answer3).all


def test_voxel_iqr_variance():

    example_values = np.zeros((3,3,3,12))
    for time_values in range(example_values.shape[-1]):
        example_values[...,time_values] = time_values

    q1, q3 = np.percentile(example_values[1,1,1,:], [25, 75])
    example_answer = np.zeros(example_values.shape[:-1]) + q3 - q1
    assert example_answer.shape == voxel_iqr_variance(example_values).shape
    assert (example_answer == voxel_iqr_variance(example_values)).all
    

TEST_FNAME = npx.fetch_file('ds114_sub009_t2r1.nii')

def test_dvars_squared():
    img = nib.load(TEST_FNAME)
    n_trs = img.shape[-1]
    n_voxels = np.prod(img.shape[:-1])
    data = img.get_fdata()
    dvals = dvars_squared(data)
    assert len(dvals) == n_trs - 1
    # Calculate the values the long way round
    data = img.get_fdata()
    prev_vol = data[..., 0]
    long_dvals = []
    for i in range(1, n_trs):
        this_vol = data[..., i]
        d = this_vol - prev_vol
        long_dvals.append(np.sum(d ** 2) / n_voxels)
        prev_vol = this_vol
    assert np.allclose(dvals, long_dvals)


