import pytest
from numpy.testing import assert_allclose

from pybaqus.reader import open_fil, read_records


@pytest.fixture
def fil_path_2d():
    return "tests/abaqus/fil/quad_CPE4.fil"


@pytest.fixture
def fil_path_3d():
    return "tests/abaqus/fil/hex_C3D8.fil"


def test_output_2D_node_set(fil_path_2d):
    """Test output for a node set.

    Parameters
    ----------
        fil_path_2d:
            The path to the *.fil file.
    """
    model = open_fil(fil_path_2d)
    displ = model.get_nodal_result(
        var="U2", step=1, inc=1, node_set="ASSEMBLY_SET_LOAD"
    )
    assert_allclose(displ, [0.1508789062499999, 0.1508789062499999])


def test_output_2D_element_set(fil_path_2d):
    """Test output for an element set.

    Parameters
    ----------
        fil_path_2d:
            The path to the *.fil file.
    """
    model = open_fil(fil_path_2d)
    displ = model.get_nodal_result(
        var="S2", step=1, inc=1, elem_set="ASSEMBLY_TEST_INSTANCE_SET-TEST_PART"
    )
    assert_allclose(displ, [1562.5, 1562.5, 1562.5, 1562.5])
