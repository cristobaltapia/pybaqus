"""
Utilities to read the ASCII *.fil files generates by abaqus.
"""
import re

import numpy as np

from .fil_result import FilResult


def open_fil(file_name):
    """Read the *.fil file

    Parameters
    ----------
    file_name : str
        Path to the *.fil file

    Returns
    -------
    Result : Object containing the results of the *.fil file

    """
    with open(file_name, "r") as result:
        lines = result.readlines()

    # Remove new lines to get one continuous string
    lines = [li.replace("\n", "") for li in lines]
    res_line = "".join(lines)

    # Make a list with each record.
    # Each record starts with `*`, so we just need to use a regex to return each
    # record.
    pattern = r"(?<=\*)+.+?(?=\*)|(?<=\*)+.+?(?=$)"
    records = re.findall(pattern, res_line)

    # Create result object
    result = FilResult(records)

    return result.model
