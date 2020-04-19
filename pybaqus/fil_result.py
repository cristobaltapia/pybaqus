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

    """

    PARSE_MAP = {
            1: ("_parse_elem_header", ),
            21: ("_parse_elem_output", ),
            101: ("_parse_nodal_output", ["U"]),
            104: ("_parse_nodal_output", ["RF"]),
            106: ("_parse_nodal_output", ["CF"]),
            107: ("_parse_nodal_output", ["COORD"]),
            146: ("_parse_nodal_output", ["TF"]),
            1900: ("_parse_element", ),
            1901: ("_parse_node", ),
            2000: ("_parse_step", ),
            }

    def __init__(self, records):
        self._records = records
        self.model = Model()

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
            + r" (\d+\.\d+(?:E|D)(?:\+|\-)\d+)|"  # floats
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
                try:
                    args = self.PARSE_MAP[key][1]
                    getattr(self, self.PARSE_MAP[key][0])(vars_i, *args)
                except:
                    getattr(self, self.PARSE_MAP[key][0])(vars_i)
            # else:
            #     print(f"Key {key} not defined!")

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
            node = Node3D(*coords, num=n_number, model=self.model)

        self.model.add_node(node)

    def _parse_elem_output(self, record):
        """Parse output data for elements.

        Parameters
        ----------
        record : list

        Returns
        -------
        TODO

        """
        # print(record)
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
        # print(record)
        pass

    def _parse_nodal_output(self, record, var):
        """Parse the nodal record

        Parameters
        ----------
        record : TODO

        Returns
        -------
        TODO

        """
        if len(record) > 4:
            for ix, r_i in enumerate(record[3:], start=1):
                self.model.add_nodal_output(node=record[2], var=f"{var}{ix}", data=r_i)
        else:
            self.model.add_nodal_output(node=record[2], var=var, data=r_i[3])

        return 1

    def _parse_step(self, record):
        """Parse the current step

        Parameters
        ----------
        record : TODO

        Returns
        -------
        TODO

        """
        n = record[7]

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

        self.model.add_step(n, data)

