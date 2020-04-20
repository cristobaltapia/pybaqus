"""
Definitions of classes that define the imported model
"""
import numpy as np
import vtk
from pyvista import UnstructuredGrid
from .step import Step


class Model:
    """Class for the model.

    This contains all the information of the model.

    """

    def __init__(self):
        self.nodes: dict = dict()
        self.elements: dict = dict()
        self.element_sets: dict = dict()
        self.node_sets: dict = dict()
        self.results: dict = dict()
        self.metadata: dict = dict()
        self.mesh = None
        self.elem_output: dict = dict()
        self.nodal_output: dict = dict()
        self.steps: dict = dict()
        self._curr_out_step: int = None
        self._curr_incr: int = None
        self._dimension: int = None

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node._num] = node

    def add_element(self, element):
        if element not in self.elements:
            self.elements[element._num] = element

    def add_set(self, name, elements, s_type):
        """Add an element set.

        Parameters
        ----------
        name : TODO

        Returns
        -------
        TODO

        """
        if s_type == "node":
            self.node_sets[name] = elements
        elif s_type == "element":
            self.element_sets[name] = elements

    def add_elem_output(self, elem, var, data, step, inc):
        """Add element output data

        Parameters
        ----------
        var : TODO
        data : TODO

        Returns
        -------
        TODO

        """
        curr_step = self._curr_out_step
        curr_inc = self._curr_incr

        if var not in self.elem_output[curr_step][curr_inc]:
            self.elem_output[curr_step][curr_inc][var] = dict()

        self.elem_output[step][inc][var][elem] = data

    def add_nodal_output(self, node, var, data, step, inc):
        """Add nodal output results

        Parameters
        ----------
        node : int
            Node to which assign the data
        var : str
            Name of the variable
        data : float
            Value of the output

        """
        curr_step = self._curr_out_step
        curr_inc = self._curr_incr

        if var not in self.nodal_output[curr_step][curr_inc]:
            self.nodal_output[curr_step][curr_inc][var] = dict()

        self.nodal_output[step][inc][var][node] = data

    def add_step(self, n, data):
        """Add a new step to the output database

        Parameters
        ----------
        n : int
            Index of the step
        data : list
            Arguments for the Step object

        Returns
        -------
        TODO

        """
        # Add step to model
        if n not in self.steps:
            self.steps[n] = Step(self, n, data)

            inc_n = data["increment number"]
            self._curr_out_step = n
            self._curr_incr = inc_n
            # Initialize output repository for the current increment in step
            self.nodal_output[n] = {inc_n: dict()}
            self.elem_output[n] = {inc_n: dict()}

        # Add increment to step
        else:
            step_time = data["step time"]
            load_prop = data["load proportionality"]
            time_inc = data["time increment"]
            inc_n = data["increment number"]

            # Initialize output repository for the current increment in step
            self.nodal_output[n][inc_n] = dict()
            self.elem_output[n][inc_n] = dict()

            self._curr_out_step = data["step number"]
            self._curr_incr = data["increment number"]

            self.steps[n].add_increment(inc_n, time_inc, step_time, load_prop)

    def get_nodal_result(self, var, step, inc):
        """Get nodal results

        Parameters
        ----------
        var : TODO

        Returns
        -------
        TODO

        """
        # FIXME: have this variable sorted globally
        keys = sorted(list(self.nodes.keys()))

        results = self.nodal_output[step][inc][var]

        list_res = [results[k] for k in keys]

        return np.array(list_res)

    def get_nodal_vector_result(self, var, step, inc):
        """Get the vector of a variable at each node.

        Parameters
        ----------
        step : TODO
        inc : TODO

        Returns
        -------
        array :
            Nx3 array of displacements in each node

        """
        nodes = self.nodes
        keys = sorted(list(self.nodes.keys()))
        coords = list()

        for k in keys:
            coords.append(self._get_node_vector_result(k, var, step, inc))

        coords_ar = np.array(coords)

        return coords_ar

    def get_element_result(self, var, step, inc):
        """Get element results.

        Parameters
        ----------
        var : TODO
        step : TODO
        inc : TODO

        Returns
        -------
        TODO

        """
        # FIXME: have this variable sorted globally
        keys = sorted(list(self.elements.keys()))
        keys_out = self.elem_output[step][inc][var].keys()

        results = self.elem_output[step][inc][var]

        list_res = [results[k] if k in keys_out else np.nan for k in keys]

        return np.array(list_res)

    def add_metadata(self, metadata):
        """Add metadata to the model."""
        self.metadata[metadata[0]] = metadata[1]

    def get_node_coords(self, node_set=None):
        """Get a list with the node coordinates.
        """
        nodes = self.nodes

        if node_set is not None:
            node_ids = self.node_sets[node_set]
            nodes = {k: nodes[k] for k in node_ids}

        keys = sorted(list(nodes.keys()))
        coords = list()

        for k in keys:
            coords.append(nodes[k].get_coords())

        coords_ar = np.array(coords)

        return coords_ar

    def get_deformed_node_coords(self, step, inc, scale=1):
        """Get deformed node coordinates.

        Parameters
        ----------
        step : int
            Step to get deformations from
        inc : int
            Index of the increment in the required step.
        scale : float
            Multiply the deformations by this number.

        Returns
        -------
        array :
            2D-Array with the node coordinates

        """
        nodes = self.nodes
        keys = sorted(list(self.nodes.keys()))
        coords = list()

        for k in keys:
            # Get the nodal displacements
            u = self._get_node_vector_result(k, step, inc)
            coords.append(nodes[k].get_coords() + u * scale)

        coords_ar = np.array(coords)

        return coords_ar

    def get_cells(self, elem_set=None):
        """Get the definition of cells for all elements.

        The format is the one required by VTK.

        Returns
        -------
        cells : array
            Cells of each elements
        offset : array
            Offset for each element
        elem_type : array
            Array with element types

        """
        elements = self.elements

        if elem_set is not None:
            elem_ids = self.element_sets[elem_set]
            elements = {k: elements[k] for k in elem_ids}

        keys = sorted(list(elements.keys()))

        cells = list()
        offset = list()
        elem_type = list()

        for el_i in keys:
            cells.extend(elements[el_i].get_cell())
            offset.append(len(elements[el_i]._nodes))
            elem_type.append(elements[el_i]._elem_type)

        ar_cells = np.array(cells)
        ar_offset = np.cumsum(np.array(offset, dtype=int)) - offset[0]
        ar_elem_type = np.array(elem_type, np.int8)

        return ar_cells, ar_offset, ar_elem_type

    def get_mesh(self, elem_set=None):
        """Construct the mesh of the finite element model

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        nodes = self.get_node_coords()
        cells, offset, elem_t = self.get_cells(elem_set)

        mesh = UnstructuredGrid(offset, cells, elem_t, nodes)

        self.mesh = mesh

        return self.mesh

    def get_deformed_mesh(self, step, inc, scale=1):
        """Construct the deformed mesh in step with scaled deformations.

        Parameters
        ----------
        step : int
            Index of the needed step
        inc : int
            Index of the increment within the step
        scale : flotat
            Scale to be applied to the deformations

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        nodes = self.get_deformed_node_coords(step, inc, scale)
        cells, offset, elem_t = self.get_cells()

        mesh = UnstructuredGrid(offset, cells, elem_t, nodes)

        self.mesh = mesh

        return self.mesh

    def _get_node_vector_result(self, n, var, step, inc):
        """Get the displacement vector of the node `n`

        Parameters
        ----------
        n : int
            The index of the node.
        step : int
            The step for which the displacement is required.
        inc : int
            The increment within the required step.

        Returns
        -------
        array :
            An array with the displacements of the node

        """
        nodal_output = self.nodal_output[step][inc]

        if self._dimension == 3:
            u = np.array(
                [
                    nodal_output[f"{var}1"][n],
                    nodal_output[f"{var}2"][n],
                    nodal_output[f"{var}3"][n],
                ]
            )
        else:
            u = np.array([nodal_output[f"{var}1"][n], nodal_output[f"{var}2"][n], 0,])

        return u


class Node:
    """Define a node in the finite element model.

    Parameters
    ----------
    x : float
    y : float
    z : float
    num : float

    """

    def __init__(self, num, model):
        self._num: int = num
        self._x: float = None
        self._y: float = None
        self._z: float = None
        self._rx: float = None
        self._ry: float = None
        self._rz: float = None
        self.model = model

    def get_coords(self):
        return self._get_coords()

    def _get_coords(self):
        print("This method needs to be redefined in subclass")
        return 1


class Node2D(Node):
    """Two-dimensional node.

    Parameters
    ----------
    x : TODO
    y : TODO
    num : TODO

    """

    def __init__(self, num, dof_map, model, *dof):
        super().__init__(num, model)

        self._x = np.float(dof[dof_map[1]])
        self._y = np.float(dof[dof_map[2]])
        self._rz = np.float(dof[dof_map[6]])
        self._num = num

    def _get_coords(self):
        return (self._x, self._y, 0)


class Node3D(Node):
    """Three-dimensional node.

    Parameters
    ----------
    num : int
        Number of the node
    dof_map : dict
        Dictionary mapping the active DOF to each nodal output position
    model : `obj`:Model
        Model to which the node belongs to.
    *dof :
        The values for all degree of freedom.

    """

    def __init__(self, num, dof_map, model, *dof):
        super().__init__(num, model)

        self._x = np.float(dof[dof_map[1]])
        self._y = np.float(dof[dof_map[2]])
        self._z = np.float(dof[dof_map[3]])
        self._rx = np.float(dof[dof_map[4]])
        self._ry = np.float(dof[dof_map[5]])
        self._rz = np.float(dof[dof_map[6]])
        self._num = num

    def _get_coords(self):
        return (self._x, self._y, self._z)


class Element:
    """Define a general element of a finite element model.

    Subclasses must be implemented for each actual element.

    Parameters
    ----------
    num : int
    model : `obj`:Model

    """

    def __init__(self, num, model):
        """TODO: to be defined.

        Parameters
        ----------
        id : TODO


        """
        self._num: int = num
        self._nodes: list = None
        self._n_nodes: int = None
        self._connectivity: list = None
        self.model = model
        self.elem_type = None

    def get_nodes(self):
        return self._nodes

    def get_nodes_coords(self):
        """Get coordinates of nodes

        Returns
        -------
        TODO

        """
        all_nodes = self.model.nodes
        nodes = [all_nodes[n].get_coords() for n in self._nodes]

        return nodes

    def get_cell(self):
        # Get nodes
        nodes = np.array(self._nodes) - 1

        cell = np.array([self._n_nodes, *nodes], dtype=int)

        return cell


class Quad(Element):
    """4-node rectangular element."""

    def __init__(self, n1, n2, n3, n4, num, model):
        super().__init__(num, model)
        self._n_nodes = 4
        self._nodes = [n1, n2, n3, n4]
        self._elem_type = vtk.VTK_QUAD


class Triangle(Element):
    """3-node triangular element."""

    def __init__(self, n1, n2, n3, num, model):
        super().__init__(num, model)
        self._n_nodes = 3
        self._nodes = [n1, n2, n3]
        self._elem_type = vtk.VTK_TRIANGLE


class Tetra(Element):
    """4 node tetrahedron elements"""

    def __init__(self, n1, n2, n3, n4, num, model):
        super().__init__(num, model)
        self._n_nodes = 4
        self._nodes = [n1, n2, n3, n4]
        self._elem_type = vtk.VTK_TETRA


class Pyramid(Element):
    """5 node pyramid element."""

    def __init__(self, n1, n2, n3, n4, n5, num, model):
        super().__init__(num, model)
        self._n_nodes = 5
        self._nodes = [n1, n2, n3, n4, n5]
        self._elem_type = vtk.VTK_PYRAMID


class Wedge(Element):
    """6 node triangular prism element."""

    def __init__(self, n1, n2, n3, n4, n5, n6, num, model):
        super().__init__(num, model)
        self._n_nodes = 6
        self._nodes = [n1, n2, n3, n4, n5, n6]
        self._elem_type = vtk.VTK_WEDGE


class Hexahedron(Element):
    """8 node brick element."""

    def __init__(self, n1, n2, n3, n4, n5, n6, n7, n8, num, model):
        super().__init__(num, model)
        self._n_nodes = 8
        self._nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
        self._elem_type = vtk.VTK_PYRAMID


class LineElement(Element):
    """2 node pyramid element."""

    def __init__(self, n1, n2, num, model):
        super().__init__(num, model)
        self._n_nodes = 2
        self._nodes = [n1, n2]
        self._elem_type = vtk.VTK_LINE


ELEMENTS = {
    # Rigid
    "R2D2": LineElement,
    "R3D3": Triangle,
    "R3D4": Quad,
    # 2D Continuum
    "S4R": Quad,
    "CPS4R": Quad,
    "CPE3": Triangle,
    "CPE3H": Triangle,
    "CPS3": Triangle,
    "CPEG3": Triangle,
    # 3D Continuum
    "C3D4": Tetra,
    "C3D4H": Tetra,
    "C3D5": Pyramid,
    "C3D6": Wedge,
    "C3D6H": Wedge,
    "C3D8": Hexahedron,
    "C3D8H": Hexahedron,
    "C3D8I": Hexahedron,
    "C3D8R": Hexahedron,
    "C3D8RH": Hexahedron,
    "C3D8RS": Hexahedron,
}

