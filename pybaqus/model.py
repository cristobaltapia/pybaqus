"""
Definitions of classes that define the imported model
"""
import numpy as np
from pyvista import UnstructuredGrid
from .step import Step
from .faces import RigidSurface, DeformableSurface, Face


class Model:
    """Class for the model.

    This contains all the information of the model.

    Attributes
    ----------
    nodes : dict
    elements : dict
    element_sets : dict
    node_sets : dict
    surfaces : dict

    """

    def __init__(self):
        self.nodes: dict = dict()
        self.elements: dict = dict()
        self.element_sets: dict = dict()
        self.node_sets: dict = dict()
        self.surfaces: dict = dict()
        self.results: dict = dict()
        self.metadata: dict = dict()
        self.mesh = None
        self.elem_output: dict = dict()
        self.nodal_output: dict = dict()
        self.steps: dict = dict()
        self._curr_out_step: int = None
        self._curr_incr: int = None
        self._dimension: int = None
        self._status: int = None

    def set_status(self, n):
        """Set the  SDV number controling the element deletion

        Parameters
        ----------
        n : TODO

        Returns
        -------
        TODO

        """
        self._status = n

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes[node._num] = node

    def add_element(self, element):
        if element not in self.elements:
            self.elements[element.num] = element

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

    def add_deformable_surface(self, name, dimension, master_surf):
        """Add a surface to the model.

        Parameters
        ----------
        name : TODO
        faces : TODO

        """
        if name not in self.surfaces:
            self.surfaces[name] = DeformableSurface(name, dimension, self, master_surf)

    def add_rigid_surface(self, name, dimension, ref_point):
        """Add a surface to the model.

        Parameters
        ----------
        name : TODO
        faces : TODO

        """
        if name not in self.surfaces:
            self.surfaces[name] = RigidSurface(name, dimension, self, ref_point)

    def add_face_to_surface(self, surface, face_info):
        """Add a face to an existing surface

        Parameters
        ----------
        surface : str
            Label of the surface to add the facelt to.
        face_info : dict
            Dictionary with the data to create a Face object.

        """
        elem_n = face_info["element"]
        if elem_n == 0:
            element = None
        else:
            element = self.elements[elem_n]

        face = Face(element, face_info["face"], face_info["nodes"])
        self.surfaces[surface].add_face(face)

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
        if var not in self.elem_output[step][inc]:
            self.elem_output[step][inc][var] = dict()

        if elem not in self.elem_output[step][inc][var]:
            self.elem_output[step][inc][var][elem] = list()

        self.elem_output[step][inc][var][elem].append(data)

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

        if var in self.nodal_output[step][inc]:
            results = self.nodal_output[step][inc][var]
        elif var in self.elem_output[step][inc]:
            results = self._nodal_result_from_elements(var, step, inc)
        else:
            # FIXME: handle errors properly some day
            print("Variable not present")

        list_res = [results[k] for k in keys]

        return np.array(list_res)

    def _nodal_result_from_elements(self, var, step, inc):
        """Get nodal results from element results by extrapolating.

        Shape functions are used to extrapolate to the nodes.

        Parameters
        ----------
        var : str
            Result variable.
        step : int
            Step
        inc : int
            Increment

        Returns
        -------
        array

        """
        keys_out = self.elem_output[step][inc][var].keys()
        output = self.elem_output[step][inc][var]

        elements = self.elements

        # FIXME: there are some hacky things here. Try to fix that
        nodes = self.nodes
        res_nodes = np.zeros(len(nodes) + 1)
        counter = np.zeros(len(nodes) + 1)

        for ix in keys_out:
            var_i = output[ix]
            # Returns extrapolated variables and respective node labels
            nodal_i, elem_nodes = elements[ix].extrapolate_to_nodes(var_i)
            res_nodes[elem_nodes] += nodal_i
            counter[elem_nodes] += 1

        # FIXME: Another hacky fix
        counter[counter == 0] = np.nan

        result = res_nodes / counter

        # FIXME: output correct size
        return result

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

        if self._status is not None:
            status = self.elem_output[step][inc][f"SDV{self._status}"]
            del_elem = [k for k, v in status.items() if v[0] == 0]
            keys_out = [k for k in keys_out if k not in del_elem]
            keys= [k for k in keys if k not in del_elem]

        results = self.elem_output[step][inc][var]

        list_res = [np.mean(results[k]) if k in keys_out else np.nan for k in keys]

        ar_results = np.array(list_res)

        return ar_results

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
            coords.append(nodes[k].coords)

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
            u = self._get_node_vector_result(k, "U", step, inc)
            coords.append(nodes[k].coords + u * scale)

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

    def get_surface(self, name, step=None, inc=None, scale=1):
        """Get mesh of surface.

        Parameters
        ----------
        name : TODO

        Returns
        -------
        mesh :
            Mesh representation of the surface.

        """
        surface = self.surfaces[name]

        return surface.mesh

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
        status : int, None
            Solution-dependent state variable that controls the element deletion

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        nodes = self.get_deformed_node_coords(step, inc, scale)

        if self._status:
            status = self.elem_output[step][inc][f"SDV{self._status}"]

        cells, offset, elem_t = self.get_cells(status=status)

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

    def post_import_actions(self):
        """Execute some functions after importing all the records into the model."""
        pass

