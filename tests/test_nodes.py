import pytest
from numpy.testing import assert_allclose

from pybaqus.reader import open_fil


@pytest.fixture
def fil_path_2d():
    return "tests/abaqus/fil/quad_CPE4.fil"


@pytest.fixture
def fil_path_3d():
    return "tests/abaqus/fil/hex_C3D8.fil"


def test_parse_node_2d(fil_path_2d):
    model = open_fil(fil_path_2d)
    n1 = model.nodes[0]
    n2 = model.nodes[1]
    n3 = model.nodes[2]
    n4 = model.nodes[3]

    assert_allclose(n1.coords, [0.1, 0.2, 0.0])
    assert_allclose(n2.coords, [12.9, 0.2, 0.0])
    assert_allclose(n3.coords, [0.1, 10.5, 0.0])
    assert_allclose(n4.coords, [12.9, 10.5, 0.0])


def test_parse_node_3d(fil_path_3d):
    model = open_fil(fil_path_3d)
    n1 = model.nodes[0]
    n2 = model.nodes[1]
    n3 = model.nodes[2]
    n4 = model.nodes[3]
    n5 = model.nodes[4]
    n6 = model.nodes[5]
    n7 = model.nodes[6]
    n8 = model.nodes[7]

    assert_allclose(n1.coords, [0.0, 0.0, 0.0])
    assert_allclose(n2.coords, [10.0, 0.0, 0.0])
    assert_allclose(n3.coords, [0.0, 20.0, 0.0])
    assert_allclose(n4.coords, [10.0, 20.0, 0.0])
    assert_allclose(n5.coords, [0.0, 0.0, 30.0])
    assert_allclose(n6.coords, [10.0, 0.0, 30.0])
    assert_allclose(n7.coords, [0.0, 20.0, 30.0])
    assert_allclose(n8.coords, [10.0, 20.0, 30.0])
