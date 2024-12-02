"""Test data for element result validation"""

import numpy as np

ELEMENT_QUAD_TYPES = [
    {
        "name": "CPE4",
        "variable": "S2",
        "expected": np.array([1.5625, 1.5625, 1.5625, 1.5625]),
    },
    {
        "name": "CPE4H",
        "variable": "S2",
        "expected": np.array([1.5625, 1.5625, 1.5625, 1.5625]),
    },
    {
        "name": "CPS4",
        "variable": "S2",
        "expected": np.array([1.5625, 1.5625, 1.5625, 1.5625]),
    },
    {
        "name": "CPS4I",
        "variable": "S2",
        "expected": np.array([1.5625, 1.5625, 1.5625, 1.5625]),
    },
    {
        "name": "CPS4R",
        "variable": "S2",
        "expected": np.array([1.5625, 1.5625, 1.5625, 1.5625]),
    },
]

# Elements with triangular shapes
ELEMENT_TRI_TYPES = [
    {
        "name": "CPE3",
        "variable": "S1",
        "expected": np.array([1.136868e-13, 1.136868e-13, 1.136868e-13]),
    },
    {
        "name": "CPE3H",
        "variable": "S1",
        "expected": np.array([-1.136868e-13, -1.136868e-13, -1.136868e-13]),
    },
    {
        "name": "CPS3",
        "variable": "S1",
        "expected": np.array([5.684342e-14, 5.684342e-14, 5.684342e-14]),
    },
]

# 3D Elements with hexagonal shapes
ELEMENT_HEX_TYPES = [
    {
        "name": "C3D8",
        "variable": "S3",
        "expected": np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]),
    },
]
