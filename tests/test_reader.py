import pytest
from numpy.testing import assert_allclose

from pybaqus.reader import open_fil, read_records


@pytest.fixture
def fil_path_2d():
    return "tests/abaqus/fil/quad_CPE4.fil"


@pytest.fixture
def fil_path_3d():
    return "tests/abaqus/fil/hex_C3D8.fil"


def test_read_records(fil_path_2d):
    records = read_records(fil_path_2d)
    assert hasattr(records, "__iter__")  # Check if it's an iterator


@pytest.mark.parametrize("progress", [False, True])
def test_open_fil(fil_path_2d, progress, capsys):
    result = open_fil(fil_path_2d, progress=progress)
    assert result is not None

    captured = capsys.readouterr()
    if progress:
        assert "Reading records..." in captured.out
        assert "Parsing records..." in captured.out
    else:
        assert captured.out == ""


def test_open_fil_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        open_fil("nonexistent_file.fil")


def test_abaqus_release(fil_path_2d):
    model = open_fil(fil_path_2d)
    # Check the release
    assert model.release["release"] == "6.23-1"


def test_heading(fil_path_2d):
    model = open_fil(fil_path_2d)
    assert model.heading == "Test elements of the type CPE4 with quad shape"


def test_model_size(fil_path_2d):
    model = open_fil(fil_path_2d)
    assert model.size["elements"] == len(model.elements)
    assert model.size["nodes"] == len(model.nodes)


def test_discontinuous_nodes():
    model = open_fil("tests/abaqus/fil/discontinuous_numbering_2D.fil")
    assert model.size["nodes"] == len(model.nodes)


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


@pytest.mark.parametrize(
    "file_path,expected_record_count",
    [
        ("tests/abaqus/fil/quad_CPE4.fil", 50),
    ],
)
def test_read_records_count(file_path, expected_record_count):
    records = list(read_records(file_path))
    assert len(records) == expected_record_count


def test_node_sets_2D(fil_path_2d):
    model = open_fil(fil_path_2d)
    node_sets = model.node_sets
    expected_sets = {
        "ASSEMBLY_SET_BC_1": [1],
        "ASSEMBLY_SET_BC_2": [2],
        "ASSEMBLY_SET_LOAD": [3, 4],
    }

    assert expected_sets.keys() == node_sets.keys()
    for k, v in expected_sets.items():
        assert v == node_sets[k]


def test_node_sets_3D(fil_path_3d):
    model = open_fil(fil_path_3d)
    node_sets = model.node_sets
    expected_sets = {
        "ASSEMBLY_SET_BC_1": [1],
        "ASSEMBLY_SET_BC_2": [4],
        "ASSEMBLY_SET_BC_3": [2, 3],
        "ASSEMBLY_SET_LOAD": [5, 6, 7, 8],
    }

    assert expected_sets.keys() == node_sets.keys()
    for k, v in expected_sets.items():
        assert v == node_sets[k]
