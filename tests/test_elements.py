import pytest
from numpy.testing import assert_allclose

from pybaqus.reader import open_fil, read_records

from test_data.element_results import ELEMENT_STRESS_CASES


@pytest.mark.parametrize("test_case", ELEMENT_STRESS_CASES, ids=lambda x: x["name"])
def test_element_stress(test_case):
    """Test element stress results for different element types.

    Args:
        test_case: Dictionary containing test parameters and expected results
    """
    model = open_fil(f"tests/abaqus/fil/quad_{test_case['name']}.fil")
    stress = model.get_nodal_result(var=test_case["variable"], step=1, inc=1)
    assert_allclose(
        stress,
        test_case["expected"],
        err_msg=f"Failed for element type: {test_case['name']}",
        rtol=1e-6,
    )
