import pytest

from pybaqus.reader import open_fil, read_records


@pytest.fixture
def fil_path():
    return "tests/abaqus/fil/quad_CPE4.fil"


def test_read_records(fil_path):
    records = read_records(fil_path)
    assert hasattr(records, "__iter__")  # Check if it's an iterator


# @pytest.mark.parametrize("progress", [False, True])
# def test_open_fil(sample_fil_path, progress, capsys):
#     result = open_fil(sample_fil_path, progress=progress)
#     assert result is not None
#
#     captured = capsys.readouterr()
#     if progress:
#         assert "Reading records..." in captured.out
#         assert "Parsing records..." in captured.out
#     else:
#         assert (
#             captured.out
#             == "Record key 1921 (Abaqus release, etc.) not yet implemented!\nRecord key 1922 (Heading) not yet implemented!\n"
#         )


def test_open_fil_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        open_fil("nonexistent_file.fil")


@pytest.mark.parametrize(
    "file_path,expected_record_count",
    [
        ("tests/abaqus/fil/quad_CPE4.fil", 50),
    ],
)
def test_read_records_count(file_path, expected_record_count):
    records = list(read_records(file_path))
    assert len(records) == expected_record_count


def test_abaqus_release(fil_path):
    model = open_fil(fil_path)
    # Check the release
    assert model.release["release"] == "6.23-1"

def test_model_size(fil_path):
    model = open_fil(fil_path)
    assert model.size["elements"] == len(model.elements)
    assert model.size["nodes"] == len(model.nodes)
