"""
Utilities to read the ASCII *.fil files generates by abaqus.
"""
import re
from tqdm import tqdm

from .fil_result import FilParser


def open_fil(file_name, progress=False):
    """Read the *.fil file

    Parameters
    ----------
    file_name : str
        Path to the *.fil file
    progress : bool
        Indicates whether the progress of the reading process should be shown
        in a progress bar. (default: False)

    Returns
    -------
    Result : Object containing the results of the *.fil file

    """
    if progress:
        print("Reading records...")

    with open(file_name, "r") as result:
        lines = result.read().splitlines()

    res_line = "".join(lines)
    del lines

    # Make a list with each record.
    # Each record starts with `*`, so we just need to use a regex to return each
    # record.
    pattern = r"(?<=\*)+.+?(?=\*)|(?<=\*)+.+?(?=$)"
    records = re.findall(pattern, res_line)
    del res_line

    # Create result object
    if progress:
        print("Parsing records...")

    result = FilParser(records, progress=progress)
    del records

    return result.model
