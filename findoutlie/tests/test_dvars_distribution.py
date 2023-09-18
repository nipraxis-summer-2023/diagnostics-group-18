""" Test script for detector functions

Run these tests with::

    python3 findoutlie/tests/test_dvars_distribution.py

or better, in IPython::

    %run findoutlie/tests/test_dvars_distribution.py

or even better, from the terminal::

    pytest findoutlie/tests/test_dvars_distribution.py

"""

import numpy as np
from findoutlie.dvars_distribution import distribution_mean, voxel_differences, voxel_iqr_variance
from scipy.special import ndtri
from math import isclose

def test_dvars_distribution():
    example_values1 = np.ones((3,3,3,6))
    example_values2 = np.zeros((3,3,3,12))
    for time_values in range(example_values2.shape[-1]):
        example_values2[...,time_values] = time_values**2
    
    example_answer2_array = voxel_differences(example_values2)[1,1,1,:]
    q1, q3 = np.percentile(example_answer2_array, [25, 75])
    answer2 = (q3 - q1)/(ndtri(0.75) - ndtri(0.25))

    assert distribution_mean(example_values1) == 0
    assert isclose(distribution_mean(example_values2), answer2)

def test_voxel_differences():
    example_values = np.ones((3,3,3,6))
    example_values1 = np.append(example_values,example_values+1, axis=3)

    # check correct shape for answer
    assert voxel_differences(example_values).shape == (3,3,3,5)

    # check correct order of operations and values
    example_answer1 = np.zeros(example_values.shape[:-1]+(example_values.shape[-1] - 1,))
    example_answer1 = np.append(example_answer1,example_values, axis=3)
    assert (voxel_differences(example_values1) == example_answer1).all

def test_voxel_iqr_variance():

    example_values = np.zeros((3,3,3,12))
    for time_values in range(example_values.shape[-1]):
        example_values[...,time_values] = time_values

    q1, q3 = np.percentile(example_values[1,1,1,:], [25, 75])
    example_answer = np.zeros(example_values.shape[:-1]) + q3 - q1
    assert example_answer.shape == voxel_iqr_variance(example_values).shape
    assert (example_answer == voxel_iqr_variance(example_values)).all
    




