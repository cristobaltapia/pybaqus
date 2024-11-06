"""Test data for element result validation"""

import numpy as np

ELEMENT_STRESS_CASES = [
    {
        "name": "CPE4",
        "variable": "S1",
        "expected": np.array([2.273737e-13, -3.105982e-13, 8.322454e-14, 2.273737e-13]),
    },
    {
        "name": "CPE4H",
        "variable": "S1",
        "expected": np.array(
            [-1.705302e-13, -9.845562e-14, 9.845596e-14, -1.705300e-13]
        ),
    },
    {
        "name": "CPS4",
        "variable": "S1",
        "expected": np.array(
            [1.213024e-13, -3.542511e-13, -1.573397e-13, 2.197581e-13]
        ),
    },
    {
        "name": "CPS4I",
        "variable": "S1",
        "expected": np.array([2.040588e-15, -3.029826e-13, 1.892958e-13, 3.958633e-13]),
    },
    {
        "name": "CPS4R",
        "variable": "S1",
        "expected": np.array([1.705303e-13, 1.705303e-13, 1.705303e-13, 1.705303e-13]),
    },
]
