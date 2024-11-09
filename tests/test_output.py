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


def test_output_3D_node_set(fil_path_3d):
    """Test output for a node set.

    Parameters
    ----------
        fil_path_2d:
            The path to the *.fil file.
    """
    model = open_fil(fil_path_3d)
    displ = model.get_nodal_result(
        var="U2", step=1, inc=1, node_set="ASSEMBLY_SET_LOAD"
    )
    assert_allclose(
        displ,
        [
            0.05991461924418786,
            0.0582193538221954,
            0.05027534615199374,
            0.0551842083097384,
        ],
    )


def test_output_3D_element_set(fil_path_3d):
    """Test output for an element set.

    Parameters
    ----------
        fil_path_3d:
            The path to the *.fil file.
    """
    model = open_fil(fil_path_3d)
    displ = model.get_nodal_result(
        var="S2", step=1, inc=1, elem_set="ASSEMBLY_TEST_INSTANCE_SET-TEST_PART"
    )
    assert_allclose(
        displ,
        [
            26.275343053264052,
            -38.063847948411876,
            78.43215874816171,
            13.046268627606647,
            -48.20076875561976,
            -16.610991953481744,
            3.9560469392777455,
            34.49912462253665,
        ],
    )
