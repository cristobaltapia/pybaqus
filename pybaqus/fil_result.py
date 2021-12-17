"""
Class for the Fil results
"""
import re

import numpy as np
from tqdm import tqdm

from .model import Model
from .nodes import Node2D, Node3D
from .elements import ELEMENTS, N_INT_PNTS


class FilParser:
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
        5: ("_parse_elem_output", ["SDV"]),
        101: ("_parse_nodal_output", ["U"]),
        104: ("_parse_nodal_output", ["RF"]),
        106: ("_parse_nodal_output", ["CF"]),
        107: ("_parse_nodal_output", ["COORD"]),
        146: ("_parse_nodal_output", ["TF"]),
        1501: ("_parse_surface", [False]),
        1502: ("_parse_surface", [True]),
        1503: ("_parse_contact_output_request", []),
        1504: ("_parse_curr_contact_node", []),
        1511: ("_parse_surface_output", ["CSTRESS"]),
        1900: ("_parse_element", []),
        1901: ("_parse_node", []),
        1902: ("_parse_active_dof", []),
        1911: ("_parse_output_request", []),
        1921: ("_parse_not_implemented", ["Abaqus release, etc."]),
        1922: ("_parse_not_implemented", ["Heading"]),
        1931: ("_parse_set", [False, "node"]),
        1932: ("_parse_set", [True, "node"]),
        1933: ("_parse_set", [False, "element"]),
        1934: ("_parse_set", [True, "element"]),
        1940: ("_parse_label_cross_ref", []),
        2000: ("_parse_step", ["start"]),
        2001: ("_parse_step", ["end"]),
    }

    CONTACT_OUT = {
        "CSTRESS": ["CPRESS", "CSHEAR1", "CSHEAR2"],
    }

    def __init__(self, records, progress):
        self._records = records
        self.model = Model()

        self._curr_elem_out: int = None
        self._curr_n_int_points: int = None
        self._curr_step: int = None
        self._curr_inc: int = None
        self._curr_loc_id: int = None
        self._flag_output: int = None
        self._curr_output_node: int = None
        self._output_request_set: str = None
        self._output_elem_type: str = None
        self._dof_map: dict = dict()
        self._model_dimension: int = None
        self._node_records: list = list()

        self._curr_set: int = None
        self._tmp_sets: dict = {"element": dict(), "node": dict()}
        self._label_cross_ref: dict = dict()
        self._curr_surface: int = None
        self._tmp_surf: dict = dict()
        self._tmp_faces: dict = dict()
        self._node_elems: dict = dict()

        self._parse_records(progress)

    def _parse_records(self, progress):
        """Parse the imported records."""
        records = self._records

        pattern = (
            r"[ADEI](?: \d(\d+)|"  # ints
            + r"((?: |-)\d+\.\d+(?:E|D)(?:\+|-)\d+)|"  # floats
            + r"(.{8}))"  # strings
        )

        # Parse each record
        for r_i in tqdm(records, disable=(not progress), leave=False, unit="record",
                        dynamic_ncols=True):
            m_rec = re.findall(pattern, r_i)

            # Get each variable
            vars_i = list(map(self._convert_record, m_rec))

            # Process record
            key = vars_i[1]
            # Lookup the key in dictionary and execute the respective functions
            if key in self.PARSE_MAP:
                args = self.PARSE_MAP[key][1]
                getattr(self, self.PARSE_MAP[key][0])(vars_i, *args)
            else:
                print(f"Key {key} not defined!")

        # Execute post-read actions on the model
        self._post_parse_all_surfaces()
        self._reference_elems_in_nodes()
        self.model.post_import_actions()

    def _convert_record(self, record):
        """Convert one record to a list of numbers and strings."""
        # For each variable three matches are made (why?), so we need to
        # take the one with the data (the only one which is not am empty
        # string)
        if record[0] != "":
            return int(record[0])
        elif record[1] != "":
            return float(record[1].replace("D", "E"))
        else:
            return record[2]

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

        # Add a reference to the node poinitng at the element
        for n in nodes:
            if n in self._node_elems.keys():
                self._node_elems[n].append(e_number)
            else:
                self._node_elems[n] = [e_number]

        ElementClass = ELEMENTS[e_type]

        element = ElementClass(*nodes, num=e_number, model=self.model, code=e_type)
        element.n_integ_points = N_INT_PNTS[e_type]
        self.model.add_element(element)

    def _parse_node(self, record):
        """Parse the data of a node

        Parameters
        ----------
        record : list

        """
        # Wait until the 'Active degree of freedom' key has been processed
        if self._model_dimension is None:
            self._node_records.append(record)
        else:
            n_number = record[2]
            dofs = record[3:]
            dof_map = self._dof_map

            if self._model_dimension == 2:
                node = Node2D(n_number, dof_map, self.model, *dofs)
            else:
                node = Node3D(n_number, dof_map, self.model, *dofs)

            self.model.add_node(node)

    def _parse_all_nodes(self):
        """Parse all nodes.

        This has to be executed after the active degree of freedom
        are specified.

        Parameters
        ----------
        records : list
            A list of all the records with nodes

        Returns
        -------
        TODO

        """
        records = self._node_records

        for record in records:
            self._parse_node(record)

        self._node_records = list()

    def _parse_elem_output(self, record, var):
        """Parse output data for elements.

        Parameters
        ----------
        record : list
        var : str
            Name of the variable

        Returns
        -------
        TODO

        """
        step = self._curr_step
        inc = self._curr_inc

        # This flags the type of output: element (0), nodal (1), modal
        # (2), or element set energy (3)
        flag_out = self._flag_output

        if flag_out == 0:
            n_elem = self._curr_elem_out
            # Get number of integration points
            int_point = self._curr_n_int_point

            # Append all the records
            for ix, data in enumerate(record[2:], start=1):
                self.model.add_elem_output(n_elem, f"{var}{ix}", data, step, inc, int_point)

        elif flag_out == 1:
            n_node = self._curr_elem_out

            for ix, data in enumerate(record[2:], start=1):
                self.model.add_nodal_output(n_node, f"{var}{ix}", data, step, inc)

        elif flag_out == 2:
            # TODO: implement modal output
            pass

        # flag_out == 3:
        else:
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
        # loc_id:
        # - 0 if the subsequent records contain data at an integration point;
        # - 1 if the subsequent records contain values at the centroid of the element;
        # - 2 if the subsequent records contain data at the nodes of the element;
        # - 3 if the subsequent records contain data associated with rebar within an element;
        # - 4 if the subsequent records contain nodal averaged values;
        # - 5 if the subsequent records contain values associated with the whole element
        loc_id = record[5]
        name_rebar = record[6]
        n_direct_stresses = record[7]
        n_shear_stresses = record[8]
        n_diretions = record[9]
        n_sec_force_comp = record[10]

        # Append the element/node number to the list of elements/nodes which
        # data is going to be read next
        self._curr_elem_out = num
        self._curr_n_int_point = n_int_point
        self._curr_loc_id = loc_id
        self._curr_int_point_data = dict()

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

        if len(record) > 2:
            for ix, r_i in enumerate(record[3:], start=1):
                self.model.add_nodal_output(node=record[2], var=f"{var}{ix}", data=r_i,
                                            step=step, inc=inc)
        else:
            self.model.add_nodal_output(node=record[0], var=var, data=record[1], step=step,
                                        inc=inc)

        return 1

    def _parse_surface_output(self, record, var):
        """Parse results from surfaces.

        Parameters
        ----------
        record : TODO
        var : str
            Name of the variable to be processed, e.g.: "CSTRESS"

        Returns
        -------
        TODO

        """
        step = self._curr_step
        inc = self._curr_inc
        node = self._curr_output_node

        for ix, comp_i in enumerate(record[2:]):
            self.model.add_nodal_output(node=node, var=self.CONTACT_OUT[var][ix],
                                        data=comp_i, step=step, inc=inc)

    def _parse_contact_output_request(self, record):
        """Parse surfaces and nodes associated to contact pair.

        Parameters
        ----------
        record : TODO

        """
        id_slave = int(record[3].strip())
        id_master = int(record[4].strip())
        name_slave = self._label_cross_ref[id_slave]
        name_master = self._label_cross_ref[id_master]
        self.model.add_contact_pair(master=name_master, slave=name_slave)

    def _parse_curr_contact_node(self, record):
        """Parse the current node associated to the surface output.

        Parameters
        ----------
        record : TODO

        """
        self._curr_output_node = record[2]
        self._no_of_components = record[3]

    def _parse_output_request(self, record):
        """Parse the output request

        Parameters
        ----------
        record : TODO

        """
        self._flag_output = record[2]
        self._output_request_set = record[3]
        if self._flag_output == 0:
            self._output_elem_type = record[4]

    def _parse_step(self, record, flag):
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

    def _parse_active_dof(self, record):
        """Parse the active degrees of freedom.

        Parameters
        ----------
        record : TODO

        Returns
        -------
        TODO

        """
        active_dof = np.array(record[2:], dtype=int)
        dimension = np.sum(np.not_equal(active_dof[:3], np.zeros(3)), dtype=int)
        self._model_dimension = dimension
        self.model._dimension = dimension

        # (k + 1): because the dof's start at 1
        # (val -1): because they will be reference to a list, which is 0-indexed
        self._dof_map = {(k + 1): (val - 1) if val != 0 else 0
                         for k, val in enumerate(active_dof)}

        # Process all nodes
        self._parse_all_nodes()

    def _parse_set(self, record, add, s_type):
        """Parse the element sets

        Parameters
        ----------
        record : TODO
        add : bool
            Flags whether records are added to an existing set a new set has to be
            created.
        s_type : str
            Type of set ("element", "node")

        Returns
        -------
        TODO

        """

        if add:
            elements = record[2:]
            ref = self._curr_set
            self._tmp_sets[s_type][ref].extend(elements)
        else:
            elements = record[3:]
            ref = int(record[2].strip())
            self._curr_set = ref
            self._tmp_sets[s_type][ref] = elements

    def _parse_surface(self, record, add_face):
        """Parse the surface records.

        Parameters
        ----------
        record : list
            Record entry from the *.fil file.
        add_face : bool
            Flag to determine whether a face has to be added to an existing surface
            (True), or whether a new surfaces has to be created (False)

        """
        if add_face:
            # Add faces to existing surface
            label = self._curr_surface

            face_info = {
                "element": record[2],
                "face": record[3],
                "nodes": record[5:],
            }
            self._tmp_faces[label].append(face_info)

        else:
            # Create new surface container
            s_type = record[4]
            name = int(record[2].strip())
            dim = record[3]
            n_faces = record[5]

            # Set temporary variable
            self._curr_surface = name
            # Start corresponding container for the surface
            self._tmp_faces[name] = list()

            if s_type == 1:
                # Deformable surface
                n_slaves = record[6]
                if n_slaves > 0:
                    master_names = record[7:]
                else:
                    master_names = []

                self._tmp_surf[name] = {
                    "dimension": dim,
                    "type": "deformable",
                    "master names": master_names,
                }

            elif s_type == 2:
                # Rigid surface
                ref_label = record[6]
                self._tmp_surf[name] = {
                    "dimension": dim,
                    "type": "rigid",
                    "reference point": ref_label,
                }

    def _post_parse_all_surfaces(self):
        """Process all the surfaces after reading all records."""
        surfaces = self._tmp_surf
        faces = self._tmp_faces
        model = self.model

        for ix, surf in surfaces.items():
            if surf["type"] == "rigid":
                # Get name
                name = self._label_cross_ref[ix]
                dim = surf["dimension"]
                ref = surf["reference point"]
                model.add_rigid_surface(name, dim, ref)

                for face_i in faces[ix]:
                    model.add_face_to_surface(name, face_i)

            elif surf["type"] == "deformable":
                # Get name
                name = self._label_cross_ref[ix]
                dim = surf["dimension"]
                master = surf["master names"]
                self.model.add_deformable_surface(name, dim, master)

                for face_i in faces[ix]:
                    model.add_face_to_surface(name, face_i)

    def _parse_label_cross_ref(self, record):
        """Parse label cross-references

        Parameters
        ----------
        record : list
            Records parsed from the *.fil file

        """
        ref = record[2]
        label = "".join(record[3:]).strip()

        self._label_cross_ref[ref] = label

        tmp_sets = self._tmp_sets

        if ref in tmp_sets["element"]:
            elements = tmp_sets["element"][ref]
            self.model.add_set(label, elements, "element")
        elif ref in tmp_sets["node"]:
            elements = tmp_sets["node"][ref]
            self.model.add_set(label, elements, "node")

    def _parse_not_implemented(self, record, r_type):
        """Helper function to deal with the not yet implemented parsers.

        Parameters
        ----------
        record : TODO
        r_type : str

        """
        tqdm.write(f"Record key {record[1]} ({r_type}) not yet implemented!")

    def _reference_elems_in_nodes(self):
        """Add a list to each node with the elements using the node."""
        model = self.model

        for ni, elems in self._node_elems.items():
            model.nodes[ni].in_elements = elems
