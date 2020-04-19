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
        self.results: dict = dict()
        self.metadata: dict = dict()
        self.mesh = None
        self.elem_output: dict = dict()
        self.nodal_output: dict = dict()
        self.steps: dict = dict()
        self._curr_out_step: int = None
        self._curr_incr: int = None

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node._num] = node

    def add_element(self, element):
        if element not in self.elements:
            self.elements[element._num] = element

    # TODO: implement
    def add_elem_output(self, var, data):
        """Add element output data

        Parameters
        ----------
        var : TODO
        data : TODO

        Returns
        -------
        TODO

        """
        pass

    def add_nodal_output(self, node, var, data, step=1, inc=1):
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
        # Add increment to step
        else:
            step_time = data["step time"]
            load_prop = data["load proportionality"]
            time_inc = data["time increment"]
            inc_n = data["increment number"]

            # Initialize output repository for the current increment in step
            self.nodal_output[n][inc_n] = dict()

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

    def add_metadata(self, metadata):
        """Add metadata to the model."""
        self.metadata[metadata[0]] = metadata[1]

    def get_node_coords(self):
        """Get a list with the node coordinates.
        """
        nodes = self.nodes
        keys = sorted(list(self.nodes.keys()))
        coords = list()

        for k in keys:
            coords.append(nodes[k].get_coords())

        coords_ar = np.array(coords)

        return coords_ar

    def get_deformed_node_coords(self, step):
        """Get deformed node coordinates.

        Parameters
        ----------
        step : int
            Step to get deformations from

        Returns
        -------
        array :
            2D-Array with the node coordinates

        """
        pass

    def get_cells(self):
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
        keys = sorted(list(self.elements.keys()))
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

    def get_mesh(self):
        """Construct the mesh of the finite element model

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        nodes = self.get_node_coords()
        cells, offset, elem_t = self.get_cells()

        mesh = UnstructuredGrid(offset, cells, elem_t, nodes)

        self.mesh = mesh

        return self.mesh

    def get_deformed_mesh(self, step, scale):
        """Construct the deformed mesh in step with scaled deformations.

        Parameters
        ----------
        step : int
            Index of the needed step
        scale : flotat
            Scale to be applied to the deformations

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        nodes = self.get_node_coords()
        cells, offset, elem_t = self.get_cells()

        mesh = UnstructuredGrid(offset, cells, elem_t, nodes)

        self.mesh = mesh

        return self.mesh


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
        """TODO: to be defined.



        """
        self._num: int = num
        self._x: float = None
        self._y: float = None
        self._z: float = None
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

    def __init__(self, x, y, num, model):
        """TODO: to be defined.


        """
        Node.__init__(self, num, model)

        self._x = np.float(x)
        self._y = np.float(y)
        self._num = num

    def _get_coords(self):
        return (self._x, self._y, 0)


class Node3D(Node):

    """Three-dimensional node.

    Parameters
    ----------
    x : TODO
    y : TODO
    z : TODO
    num : TODO

    """

    def __init__(self, x, y, z, num, model):
        """TODO: to be defined.


        """
        Node.__init__(self, num, model)

        self._x = np.float(x)
        self._y = np.float(y)
        self._z = np.float(z)
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

    def _get_cell_plane4(self):
        """Generate cell from the node information.
        """
        # Get nodes
        nodes = np.array(self._nodes) - 1

        cell = np.array([len(nodes), *nodes], dtype=int)

        return cell


class ES4R(Element):
    """Element S4R"""

    def __init__(self, n1, n2, n3, n4, num, model):
        Element.__init__(self, num, model)
        self._n_nodes = 4
        self._nodes = [n1, n2, n3, n4]
        self._connectivity = [0, 1, 2, 3]
        self._elem_type = vtk.VTK_QUAD

    def get_cell(self):
        return self._get_cell_plane4()


class ECPS4R(Element):
    """Element S4R"""

    def __init__(self, n1, n2, n3, n4, num, model):
        Element.__init__(self, num, model)
        self._n_nodes = 4
        self._nodes = [n1, n2, n3, n4]
        self._connectivity = [0, 1, 2, 3]
        self._elem_type = vtk.VTK_QUAD

    def get_cell(self):
        return self._get_cell_plane4()


ELEMENTS = {
    "S4R": ES4R,
    "CPS4R": ECPS4R,
}

