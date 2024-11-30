import pytest
from numpy.testing import assert_allclose

from pybaqus.reader import open_fil


def test_node_sets_2D(fil_path_2d):
    model = open_fil(fil_path_2d)
    node_sets = model.node_sets
    expected_sets = {
        "ASSEMBLY_SET_BC_1": [0],
        "ASSEMBLY_SET_BC_2": [1],
        "ASSEMBLY_SET_LOAD": [2, 3],
    }

    assert expected_sets.keys() == node_sets.keys()
    for k, v in expected_sets.items():
        assert v == node_sets[k]


def test_node_sets_3D(fil_path_3d):
    model = open_fil(fil_path_3d)
    node_sets = model.node_sets
    expected_sets = {
        "ASSEMBLY_SET_BC_1": [0],
        "ASSEMBLY_SET_BC_2": [3],
        "ASSEMBLY_SET_BC_3": [1, 2],
        "ASSEMBLY_SET_LOAD": [4, 5, 6, 7],
    }

    assert expected_sets.keys() == node_sets.keys()
    for k, v in expected_sets.items():
        assert v == node_sets[k]
