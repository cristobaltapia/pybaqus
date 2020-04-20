"""
Class for the Fil results
"""
import re

import numpy as np

from .model import ELEMENTS, Model, Node2D, Node3D


class FilResult:
    """
    Parse and store the data from a *.fil file.

    Parameters
    ----------
    records : list(str)
        List of the imported records of the *.fil file

    Attributes
    ----------
    _elem_out_list : list
        List of element/node numbers that correspond to the following output records

    """

    PARSE_MAP = {
        1: ("_parse_elem_header", []),
        8: ("_parse_elem_output", ["COORD"]),
        11: ("_parse_elem_output", ["S"]),
        21: ("_parse_elem_output", ["E"]),
        101: ("_parse_nodal_output", ["U"]),
        104: ("_parse_nodal_output", ["RF"]),
        106: ("_parse_nodal_output", ["CF"]),
        107: ("_parse_nodal_output", ["COORD"]),
        146: ("_parse_nodal_output", ["TF"]),
        1501: ("_parse_not_implemented", ["Surface definition header"]),
        1502: ("_parse_not_implemented", ["Surface facet"]),
        1900: ("_parse_element", []),
        1901: ("_parse_node", []),
        1902: ("_parse_not_implemented", ["Active degrees of freedom"]),
        1911: ("_parse_output_request", []),
        1921: ("_parse_not_implemented", ["Abaqus release, etc."]),
        1922: ("_parse_not_implemented", ["Heading"]),
        1931: ("_parse_not_implemented", ["Node set"]),
        1932: ("_parse_not_implemented", ["Node set continuation"]),
        1933: ("_parse_not_implemented", ["Element set"]),
        1934: ("_parse_not_implemented", ["Element set continuation"]),
        1940: ("_parse_not_implemented", ["Label cross-reference"]),
        2000: ("_parse_step", ["start"]),
        2001: ("_parse_step", ["end"]),
    }

    def __init__(self, records):
        self._records = records
        self.model = Model()

        self._curr_elem_out: int = None
        self._curr_int_point: int = None
        self._curr_step: int = None
        self._curr_inc: int = None
        self._flag_output: int = None
        self._output_request_set: str = None
        self._output_elem_type: str = None

        self._parse_records()

    def _parse_records(self):
        """Parse the imported records.

        Returns
        -------
        TODO

        """
        records = self._records

        pattern = (
            r"[ADEI](?: \d(\d+)|"  # ints
            + r"((?: |-)\d+\.\d+(?:E|D)(?:\+|-)\d+)|"  # floats
            + r"(.{8}))"  # strings
        )

        # Parse each record
        for r_i in records:
            m_rec = re.findall(pattern, r_i)

            # Get each variable
            vars_i = list()

            for v_i in m_rec:
                # For each variable three matches are made (why?), so we need to
                # take the one with the data (the only one which is not am empty
                # string)
                if v_i[0] != "":
                    vars_i.append(int(v_i[0]))
                elif v_i[1] != "":
                    vars_i.append(float(v_i[1].replace("D", "E")))
                else:
                    vars_i.append(v_i[2])

            # Process record
            key = vars_i[1]
            # Lookup the key in dictionary and execute the respective functions
            if key in self.PARSE_MAP:
                args = self.PARSE_MAP[key][1]
                getattr(self, self.PARSE_MAP[key][0])(vars_i, *args)
            else:
                print(f"Key {key} not defined!")

    def _parse_element(self, record):
        """Parse the data of an element

        Parameters
        ----------
        record : list

        """
        # Element type
        e_type = record[3].strip()
        e_number = record[2]
        nodes = record[4:]

        ElementClass = ELEMENTS[e_type]

        element = ElementClass(*nodes, num=e_number, model=self.model)
        self.model.add_element(element)

    def _parse_node(self, record):
        """Parse the data of a node

        Parameters
        ----------
        record : list

        """
        n_number = record[2]
        coords = record[3:]

        if len(coords) == 2:
            node = Node2D(*coords, num=n_number, model=self.model)
        else:
            node = Node3D(*coords[:3], num=n_number, model=self.model)

        self.model.add_node(node)

    def _parse_elem_output(self, record, var):
        """Parse output data for elements.

        Parameters
        ----------
        record : list
        var : str
            Name of the variable
        which : str
            Correspond the data to an element or node ("elem", "node")

        Returns
        -------
        TODO

        """
        step = self._curr_step
        inc = self._curr_inc

        flag_out = self._flag_output

        if flag_out == 0:
            n_elem = self._curr_elem_out

            for ix, data in enumerate(record[2:], start=1):
                self.model.add_elem_output(n_elem, f"{var}{ix}", data, step, inc)
        elif flag_out == 1:
            n_node = self._curr_elem_out

            for ix, data in enumerate(record[2:], start=1):
                self.model.add_nodal_output(n_node, f"{var}{ix}", data, step, inc)
        elif flag_out == 2:
            # TODO: implement modal output
            pass
        elif flag_out == 3:
            # TODO: implement set energy output
            pass

    def _parse_elem_header(self, record):
        """Parse the element record

        Parameters
        ----------
        record : TODO

        Returns
        -------
        TODO

        """
        num = record[2]
        n_int_point = record[3]
        n_sec_point = record[4]
        loc_id = record[5]
        name_rebar = record[6]
        n_direct_stresses = record[7]
        n_shear_stresses = record[8]
        n_diretions = record[9]
        n_sec_force_comp = record[10]

        # Append the element/node number to the list of elements/nodes which
        # data is going to be read next
        self._curr_elem_out = num
        self._curr_int_point = n_int_point

    def _parse_nodal_output(self, record, var):
        """Parse the nodal record

        Parameters
        ----------
        record : TODO

        Returns
        -------
        TODO

        """
        step = self._curr_step
        inc = self._curr_inc

        if len(record) > 4:
            for ix, r_i in enumerate(record[3:], start=1):
                self.model.add_nodal_output(
                    node=record[2], var=f"{var}{ix}", data=r_i, step=step, inc=inc
                )
        else:
            self.model.add_nodal_output(
                node=record[2], var=var, data=r_i[3], step=step, inc=inc
            )

        return 1

    def _parse_output_request(self, record):
        """Parse the output request

        Parameters
        ----------
        record : TODO

        Returns
        -------
        TODO

        """
        self._flag_output = record[2]
        self._output_request_set = record[3]
        if self._flag_output == 0:
            self._output_elem_type = record[4]


    def _parse_step(self, record):
        """Parse the current step

        Parameters
        ----------
        record : TODO

        Returns
        -------
        TODO

        """
        if flag == "start":
            n_step = record[7]
            n_inc = record[8]

            data = {
                "total time": record[2],
                "step time": record[3],
                "max creep": record[4],
                "solution amplitude": record[5],
                "procedure type": record[6],
                "step number": record[7],
                "increment number": record[8],
                "linear perturbation": record[9],
                "load proportionality": record[10],
                "frequency": record[11],
                "time increment": record[12],
                "subheading": "".join(record[13:]),
            }

            self.model.add_step(n_step, data)

            self._curr_step = n_step
            self._curr_inc = n_inc
        else:
            self._curr_step = None
            self._curr_inc = None

    def _parse_not_implemented(self, record, r_type):
        """Helper function to deal with the not yet implemented parsers.

        Parameters
        ----------
        record : TODO
        r_type : str

        """
        print(f"Record key {record[1]} ({r_type}) not yet implemented!")
