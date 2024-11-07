"""
Class for the Fil results
see ABAQUS Analysis User's Manual. FILE OUTPUT FORMAT (ANALYSIS_1.pdf)
"""
import re
import logging

import numpy as np
from tqdm import tqdm

from .model import Model
from .nodes import Node2D, Node3D
from .elements import ELEMENTS, N_INT_PNTS

_log = logging.getLogger(__name__)


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
        2: ("_parse_elem_output", ["TEMP"]),
        3: ("_parse_elem_output", ["LOADS"]),
        4: ("_parse_elem_output", ["FLUXS"]),
        5: ("_parse_elem_output", ["SDV"]),
        6: ("_parse_elem_output", ["VOIDR"]),
        7: ("_parse_elem_output", ["FOUND"]),
        8: ("_parse_elem_output", ["COORD"]),
        9: ("_parse_elem_output", ["FV"]),
        10: ("_parse_elem_output", ["NFLUX"]),
        11: ("_parse_elem_output", ["S"]),
        12: ("_parse_elem_output", ["SINV"]),
        13: ("_parse_elem_output", ["SF"]),
        14: ("_parse_elem_output", ["ENER"]),
        15: ("_parse_elem_output", ["NFORC"]),
        16: ("_parse_elem_output", ["MSS"]),
        17: ("_parse_elem_output", ["JK"]),
        18: ("_parse_elem_output", ["POR"]),
        19: ("_parse_elem_output", ["ELEN"]),
        21: ("_parse_elem_output", ["E"]),
        22: ("_parse_elem_output", ["PE"]),
        23: ("_parse_elem_output", ["CE"]),
        24: ("_parse_elem_output", ["IE"]),
        25: ("_parse_elem_output", ["EE"]),
        26: ("_parse_elem_output", ["CRACK"]),
        27: ("_parse_elem_output", ["STH"]),
        28: ("_parse_elem_output", ["HFL"]),
        29: ("_parse_elem_output", ["SE"]),
        30: ("_parse_elem_output", ["DG"]),
        31: ("_parse_elem_output", ["CONF"]),
        32: ("_parse_elem_output", ["SJP"]),
        33: ("_parse_elem_output", ["FILM"]),
        34: ("_parse_elem_output", ["RAD"]),
        35: ("_parse_elem_output", ["SAT"]),
        36: ("_parse_elem_output", ["SS"]),
        38: ("_parse_elem_output", ["CONC"]),
        39: ("_parse_elem_output", ["MFL"]),
        40: ("_parse_elem_output", ["GELVR"]),
        42: ("_parse_elem_output", ["SPE"]),
        43: ("_parse_elem_output", ["FLUVR"]),
        44: ("_parse_elem_output", ["CFAILURE"]),
        45: ("_parse_elem_output", ["PEQC"]),
        46: ("_parse_elem_output", ["PHEPG"]),
        47: ("_parse_elem_output", ["SEPE"]),
        48: ("_parse_elem_output", ["TSHR"]),
        49: ("_parse_elem_output", ["PHEFL"]),
        50: ("_parse_elem_output", ["EPG"]),
        51: ("_parse_elem_output", ["EFLX"]),
        52: ("_parse_elem_output", ["XC"]),
        53: ("_parse_elem_output", ["UC"]),
        54: ("_parse_elem_output", ["VC"]),
        55: ("_parse_elem_output", ["HC"]),
        56: ("_parse_elem_output", ["HO"]),
        57: ("_parse_elem_output", ["RI"]),
        58: ("_parse_elem_output", ["MASS"]),
        59: ("_parse_elem_output", ["VOL"]),
        60: ("_parse_elem_output", ["CHRGS"]),
        61: ("_parse_elem_output", ["STATUS"]),
        62: ("_parse_elem_output", ["PHS"]),
        63: ("_parse_elem_output", ["RS"]),
        65: ("_parse_elem_output", ["PHE"]),
        66: ("_parse_elem_output", ["RE"]),
        73: ("_parse_elem_output", ["PEEQ"]),
        74: ("_parse_elem_output", ["PRESS"]),
        75: ("_parse_elem_output", ["MISES"]),
        76: ("_parse_elem_output", ["VOLC"]),
        77: ("_parse_elem_output", ["SVOL"]),
        78: ("_parse_elem_output", ["EVOL"]),
        79: ("_parse_elem_output", ["RATIO"]),
        80: ("_parse_elem_output", ["AMPCU"]),
        83: ("_parse_elem_output", ["SSAVG"]),
        85: ("_parse_local_csys", []),
        86: ("_parse_elem_output", ["ALPHA"]),
        87: ("_parse_elem_output", ["UVARM"]),
        88: ("_parse_elem_output", ["THE"]),
        89: ("_parse_elem_output", ["LE"]),
        90: ("_parse_elem_output", ["NE"]),
        91: ("_parse_elem_output", ["ER"]),
        94: ("_parse_elem_output", ["PHMFL"]),
        95: ("_parse_elem_output", ["PHMFT"]),
        96: ("_parse_elem_output", ["MFLT"]),
        97: ("_parse_elem_output", ["FLVEL"]),
        101: ("_parse_nodal_output", ["U"]),
        102: ("_parse_nodal_output", ["V"]),
        103: ("_parse_nodal_output", ["A"]),
        104: ("_parse_nodal_output", ["RF"]),
        105: ("_parse_nodal_output", ["EPOT"]),
        106: ("_parse_nodal_output", ["CF"]),
        107: ("_parse_nodal_output", ["COORD"]),
        108: ("_parse_nodal_output", ["POR"]),
        109: ("_parse_nodal_output", ["RVF"]),
        110: ("_parse_nodal_output", ["RVT"]),
        111: ("_parse_nodal_output", ["PU"]),
        112: ("_parse_nodal_output", ["PTU"]),
        113: ("_parse_nodal_output", ["TU"]),
        114: ("_parse_nodal_output", ["TV"]),
        115: ("_parse_nodal_output", ["TA"]),
        116: ("_parse_nodal_output", ["PPOR"]),
        117: ("_parse_nodal_output", ["PHPOT"]),
        118: ("_parse_nodal_output", ["PHCHG"]),
        119: ("_parse_nodal_output", ["RCHG"]),
        120: ("_parse_nodal_output", ["CECHG"]),
        123: ("_parse_nodal_output", ["RU"]),
        124: ("_parse_nodal_output", ["RTU"]),
        127: ("_parse_nodal_output", ["RV"]),
        128: ("_parse_nodal_output", ["RTV"]),
        131: ("_parse_nodal_output", ["RA"]),
        131: ("_parse_nodal_output", ["RA"]),
        132: ("_parse_nodal_output", ["RTA"]),
        134: ("_parse_nodal_output", ["RRF"]),
        135: ("_parse_nodal_output", ["PRF"]),
        136: ("_parse_nodal_output", ["PCAV"]),
        137: ("_parse_nodal_output", ["CVOL"]),
        138: ("_parse_nodal_output", ["RECUR"]),
        139: ("_parse_nodal_output", ["CECUR"]),
        145: ("_parse_nodal_output", ["VF"]),
        146: ("_parse_nodal_output", ["TF"]),
        151: ("_parse_nodal_output", ["PABS"]),
        201: ("_parse_nodal_output", ["NT"]),
        204: ("_parse_nodal_output", ["RFL"]),
        206: ("_parse_nodal_output", ["CFL"]),
        214: ("_parse_nodal_output", ["RFLE"]),
        221: ("_parse_nodal_output", ["NNC"]),
        237: ("_parse_nodal_output", ["MOT"]),
        264: ("_parse_nodal_output", ["VOLC"]),
        301: ("_parse_nodal_output", ["GU"]),
        302: ("_parse_nodal_output", ["GV"]),
        303: ("_parse_nodal_output", ["GA"]),
        304: ("_parse_nodal_output", ["BM"]),
        305: ("_parse_nodal_output", ["GPU"]),
        306: ("_parse_nodal_output", ["GPV"]),
        307: ("_parse_nodal_output", ["GPA"]),
        308: ("_parse_nodal_output", ["SNE"]),
        309: ("_parse_nodal_output", ["KE"]),
        310: ("_parse_nodal_output", ["T"]),
        320: ("_parse_nodal_output", ["CFF"]),
        401: ("_parse_elem_output", ["SP"]),
        402: ("_parse_elem_output", ["ALPHAP"]),
        403: ("_parse_elem_output", ["EP"]),
        404: ("_parse_elem_output", ["NEP"]),
        405: ("_parse_elem_output", ["LEP"]),
        406: ("_parse_elem_output", ["ERP"]),
        407: ("_parse_elem_output", ["DGP"]),
        408: ("_parse_elem_output", ["EEP"]),
        409: ("_parse_elem_output", ["IEP"]),
        410: ("_parse_elem_output", ["THEP"]),
        411: ("_parse_elem_output", ["PEP"]),
        412: ("_parse_elem_output", ["CEP"]),
        413: ("_parse_elem_output", ["VVF"]),
        414: ("_parse_elem_output", ["VVFG"]),
        415: ("_parse_elem_output", ["VVFN"]),
        416: ("_parse_elem_output", ["RD"]),
        421: ("_parse_elem_output", ["CKE"]),
        422: ("_parse_elem_output", ["CKLE"]),
        423: ("_parse_elem_output", ["CKLS"]),
        424: ("_parse_elem_output", ["CKSTAT"]),
        425: ("_parse_elem_output", ["ECD"]),
        426: ("_parse_elem_output", ["ECURS"]),
        427: ("_parse_elem_output", ["NCURS"]),
        441: ("_parse_elem_output", ["CKEMAG"]),
        442: ("_parse_elem_output", ["RBFOR"]),
        443: ("_parse_elem_output", ["RBANG"]),
        444: ("_parse_elem_output", ["RBROT"]),
        445: ("_parse_elem_output", ["MFR"]),
        446: ("_parse_elem_output", ["ISOL"]),
        447: ("_parse_elem_output", ["ESOL"]),
        448: ("_parse_elem_output", ["SOL"]),
        449: ("_parse_elem_output", ["ESF1"]),
        462: ("_parse_elem_output", ["SEE"]),
        463: ("_parse_elem_output", ["SEP"]),
        464: ("_parse_elem_output", ["SALPHA"]),
        473: ("_parse_elem_output", ["PEEQT"]),
        475: ("_parse_elem_output", ["CS11"]),
        476: ("_parse_elem_output", ["EMSF"]),
        477: ("_parse_elem_output", ["EDT"]),
        495: ("_parse_elem_output", ["CTF"]),
        496: ("_parse_elem_output", ["CEF"]),
        497: ("_parse_elem_output", ["CVF"]),
        498: ("_parse_elem_output", ["CSF"]),
        499: ("_parse_elem_output", ["CSLST"]),
        500: ("_parse_elem_output", ["CRF"]),
        501: ("_parse_elem_output", ["CCF"]),
        502: ("_parse_elem_output", ["CP"]),
        503: ("_parse_elem_output", ["CU"]),
        504: ("_parse_elem_output", ["CCU"]),
        505: ("_parse_elem_output", ["CV"]),
        506: ("_parse_elem_output", ["CA"]),
        507: ("_parse_elem_output", ["CFAILST"]),
        508: ("_parse_elem_output", ["PHCTF"]),
        509: ("_parse_elem_output", ["PHCEF"]),
        510: ("_parse_elem_output", ["PHCVF"]),
        511: ("_parse_elem_output", ["PHCRF"]),
        512: ("_parse_elem_output", ["PHCU"]),
        513: ("_parse_elem_output", ["PHCCU"]),
        514: ("_parse_elem_output", ["RCTF"]),
        515: ("_parse_elem_output", ["RCEF"]),
        516: ("_parse_elem_output", ["RCVF"]),
        517: ("_parse_elem_output", ["RCRF"]),
        518: ("_parse_elem_output", ["RCU"]),
        519: ("_parse_elem_output", ["RCCU"]),
        520: ("_parse_elem_output", ["PHCSF"]),
        521: ("_parse_elem_output", ["RCSF"]),
        522: ("_parse_elem_output", ["PHCV"]),
        523: ("_parse_elem_output", ["PHCA"]),
        524: ("_parse_elem_output", ["VS"]),
        525: ("_parse_elem_output", ["PS"]),
        526: ("_parse_elem_output", ["VE"]),
        542: ("_parse_elem_output", ["CNF"]),
        543: ("_parse_elem_output", ["PHCNF"]),
        544: ("_parse_elem_output", ["RCNF"]),
        546: ("_parse_elem_output", ["CIVC"]),
        547: ("_parse_elem_output", ["PHCIVSL"]),
        548: ("_parse_elem_output", ["CASU"]),
        556: ("_parse_elem_output", ["CUE"]),
        557: ("_parse_elem_output", ["CUP"]),
        558: ("_parse_elem_output", ["CUPEQ"]),
        559: ("_parse_elem_output", ["CDMG"]),
        560: ("_parse_elem_output", ["CDIF"]),
        561: ("_parse_elem_output", ["CDIM"]),
        562: ("_parse_elem_output", ["CDIP"]),
        563: ("_parse_elem_output", ["CALPHAF"]),
        1501: ("_parse_surface", [False]),
        1502: ("_parse_surface", [True]),
        1503: ("_parse_contact_output_request", []),
        1504: ("_parse_curr_contact_node", []),
        1511: ("_parse_surface_output", ["CSTRESS"]),
        1512: ("_parse_surface_output", ["CDSTRESS"]),
        1521: ("_parse_surface_output", ["CDISP"]),
        1522: ("_parse_surface_output", ["CFN"]),
        1523: ("_parse_surface_output", ["CFS"]),
        1524: ("_parse_surface_output", ["CAREA"]),
        1526: ("_parse_surface_output", ["CMN"]),
        1527: ("_parse_surface_output", ["CMS"]),
        1528: ("_parse_surface_output", ["HFL"]),
        1529: ("_parse_surface_output", ["HFLA"]),
        1530: ("_parse_surface_output", ["HFL"]),
        1531: ("_parse_surface_output", ["HTLA"]),
        1532: ("_parse_surface_output", ["SFDR"]),
        1533: ("_parse_surface_output", ["SFDRA"]),
        1534: ("_parse_surface_output", ["SFDRT"]),
        1535: ("_parse_surface_output", ["SFDRTA"]),
        1536: ("_parse_surface_output", ["WEIGHT"]),
        1537: ("_parse_surface_output", ["SJD"]),
        1538: ("_parse_surface_output", ["SJDA"]),
        1539: ("_parse_surface_output", ["SJDT"]),
        1540: ("_parse_surface_output", ["SJDTA"]),
        1541: ("_parse_surface_output", ["ECD"]),
        1542: ("_parse_surface_output", ["ECDA"]),
        1543: ("_parse_surface_output", ["ECDT"]),
        1544: ("_parse_surface_output", ["ECDTA"]),
        1545: ("_parse_surface_output", ["PFL"]),
        1546: ("_parse_surface_output", ["PFLA"]),
        1547: ("_parse_surface_output", ["PTL"]),
        1548: ("_parse_surface_output", ["PTLA"]),
        1549: ("_parse_surface_output", ["TPFL"]),
        1550: ("_parse_surface_output", ["TPTL"]),
        1570: ("_parse_surface_output", ["DBT"]),
        1571: ("_parse_surface_output", ["DBSF"]),
        1572: ("_parse_surface_output", ["DBS"]),
        1573: ("_parse_surface_output", ["XN"]),
        1574: ("_parse_surface_output", ["XS"]),
        1575: ("_parse_surface_output", ["CFT"]),
        1576: ("_parse_surface_output", ["CMT"]),
        1577: ("_parse_surface_output", ["XT"]),
        1578: ("_parse_surface_output", ["CTRQ"]),
        1592: ("_parse_surface_output", ["CPPRESSTRQ"]),
        1900: ("_parse_element", []),
        1901: ("_parse_node", []),
        1902: ("_parse_active_dof", []),
        1911: ("_parse_output_request", []),
        1921: ("_parse_abaqus_release", []),
        1922: ("_parse_heading", []),
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

        # Keep track of the current node
        self._curr_node: int = 0
        self._node_map: dict[int, int] = dict()
        self._curr_set: list = []
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

        pattern = re.compile(
            r"(?:I(?: \d(\d+))|"  # ints with I prefix
            + r"[ED]((?: |-)\d+\.\d+(?:E|D)(?:\+|-)\d+)|"  # floats with E/D prefix
            + r"A(.{8}))"  # strings with A prefix
        )

        # Parse each record
        for r_i in tqdm(records, disable=(not progress), leave=False, unit="record",
                        dynamic_ncols=True):
            m_rec = pattern.findall(r_i)

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
        self._map_node_indices_to_elements()
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
        if e_type in ELEMENTS.keys():
            ElementClass = ELEMENTS[e_type]
        else:
            _log.warning(f"Element type {e_type} not supported yet. Skipping.")
            return

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
            self._node_map[n_number] = self._curr_node
            dofs = record[3:]
            dof_map = self._dof_map

            if self._model_dimension == 2:
                node = Node2D(self._curr_node, dof_map, self.model, *dofs)
            else:
                node = Node3D(self._curr_node, dof_map, self.model, *dofs)

            self.model.add_node(node)
            self._curr_node += 1

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
            node_ix = self._node_map[record[2]]
            for ix, r_i in enumerate(record[3:], start=1):
                self.model.add_nodal_output(node=node_ix, var=f"{var}{ix}", data=r_i,
                                            step=step, inc=inc)
        else:
            node_ix = self._node_map[record[0]]
            self.model.add_nodal_output(node=node_ix, var=var, data=record[1], step=step,
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
        active_dof = np.asarray(record[2:], dtype=int)
        # Determine the dimension of the model
        dimension = np.sum(np.not_equal(active_dof[:3], np.zeros(3)), dtype=int)
        self._model_dimension = dimension
        self.model._dimension = dimension

        # (k + 1): because the dof's start at 1
        # (val - 1): because they will be referenced to a list, which is 0-indexed
        # self._dof_map = {(k + 1): (val - 1) if val != 0 else 0
        #                  for k, val in enumerate(active_dof)}
        self._dof_map = active_dof

        # Process all nodes
        self._parse_all_nodes()

    def _parse_set(self, record, add, s_type):
        """Parse the element sets

        Parameters
        ----------
        record : TODO
        add : bool
            Flags whether records are added to an existing set or a new set has to be
            created.
        s_type : str
            Type of set ("element", "node")

        """
        if add:
            elements = record[2:]
            ref = self._curr_set
            self._curr_set.extend(elements)
        else:
            elements = record[3:]
            # If the name of the set is longer than 8 chars, then an integer
            # identifier is given
            label = record[2].strip()

            try:
                int(label)
                integer_label = True
            except:
                integer_label = False

            # If we have an integer identifier, then we add the elements to
            # a temporary dictionary, which is cross-referenced with the real
            # label later (processing of record 1940).
            if integer_label:
                ref = int(label)
                self._tmp_sets[s_type][ref] = elements
                self._curr_set = self._tmp_sets[s_type][ref]
            # Otherwise we create a new set and use it directly.
            else:
                label = record[2].strip()
                curr_set = self.model.add_set(label, elements, s_type)
                self._curr_set = curr_set

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

    def _parse_local_csys(self, record):
        """Parse the local coordinate systems if available.

        Parameters
        ----------
        record : list
            Records parsed from the *.fil file

        """
        # Get element being currently processed
        curr_elem = self._curr_elem_out

        # The local coordinate system is saved for each increment, so we need
        # to get the current step and increments.
        curr_step = self.model._curr_out_step
        curr_inc = self.model._curr_incr

        if self._model_dimension == 3:
            dir_1 = np.asarray(record[2:5], dtype=float)
            dir_2 = np.asarray(record[5:], dtype=float)
            # Only the first two directions are given, but we can compute the
            # third direction with a cross-prouct.
            dir_3 = np.cross(dir_1, dir_2)
            # Stack all directions into one array
            csys = np.vstack((dir_1, dir_2, dir_3))
            self.model.add_local_csys(curr_elem, csys, step=curr_step, inc=curr_inc)

        elif self._model_dimension == 2:
            dir_1 = np.asarray(record[2:4], dtype=float)
            dir_2 = np.asarray(record[4:], dtype=float)
            # Stack all directions into one array
            csys = np.vstack((dir_1, dir_2))
            self.model.add_local_csys(curr_elem, csys, step=curr_step, inc=curr_inc)

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
        # Get reference number of the set
        ref = record[2]
        # Get name of the set
        label = "".join(record[3:]).strip()

        self._label_cross_ref[ref] = label

        tmp_sets = self._tmp_sets

        if ref in tmp_sets["element"]:
            elements = tmp_sets["element"][ref]
            self.model.add_set(label, elements, "element")
        elif ref in tmp_sets["node"]:
            elements = tmp_sets["node"][ref]
            self.model.add_set(label, elements, "node")

    def _parse_abaqus_release(self, record):
        release = record[2].strip()
        date = (record[3] + record[4]).strip()
        time = record[5].strip()
        elen = record[8]

        self.model.add_release_info(release, date, time)
        self.model.size = (record[6], record[7])
        self.model.elen = elen

    def _parse_heading(self, record):
        heading = "".join(record[2:]).strip()
        self.model.heading = heading

    def _parse_not_implemented(self, record, r_type):
        """Helper function to deal with the not yet implemented parsers.

        Parameters
        ----------
        record : TODO
        r_type : str

        """
        #tqdm.write(f"Record key {record[1]} ({r_type}) not yet implemented!")
        print(f"Record key {record[1]} ({r_type}) not yet implemented!")

    def _reference_elems_in_nodes(self):
        """Add a list to each node with the elements using the node."""
        model = self.model

        for ni, elems in self._node_elems.items():
            model.nodes[self._node_map[ni]].in_elements = elems

    def _map_node_indices_to_elements(self):
        """Map the new node indices to the elements."""
        for _, e in self.model.elements.items():
            e._nodes = [self._node_map[n] for n in e._nodes]
