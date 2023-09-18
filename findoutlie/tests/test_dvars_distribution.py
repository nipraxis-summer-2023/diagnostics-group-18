""" Test script for detector functions

Run these tests with::

    python3 findoutlie/tests/test_dvars_distribution.py

or better, in IPython::

    %run findoutlie/tests/test_dvars_distribution.py

or even better, from the terminal::

    pytest findoutlie/tests/test_dvars_distribution.py

"""

import numpy as np
from findoutlie.dvars_distribution import distribution_mean

def test_dvars_distribution():
    example_values1 = np.ones((3,3,3,6))
    example_values2 = np.append(example_values1,example_values1+1, axis=3)

    # Values for exmple 2
    example2_intensity_sum = 27*6 + 27*6*2
    example2_variance = 0.25
    example2_variance_sum = 0.25*27
    example_distr_mean2 = example2_variance_sum/example2_intensity_sum

    dvars_mean1 = distribution_mean(example_values1)
    dvars_mean2 = distribution_mean(example_values2)

    assert dvars_mean1 == 0
    assert dvars_mean2 == example_distr_mean2

