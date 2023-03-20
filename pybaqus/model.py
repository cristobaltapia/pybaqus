"""
Definitions of classes that define the imported model
"""
import numpy as np
from pyvista import UnstructuredGrid
from .step import Step
from .faces import RigidSurface, DeformableSurface, Face
from .elements import N_INT_PNTS


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
        self.contact_pairs: list = list()
        self.metadata: dict = dict()
        self.mesh = None
        self.elem_output: dict = dict()
        self.nodal_output: dict = dict()
        self.local_csys: dict = dict()
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
        self.nodes[node._num] = node

    def add_element(self, element):
        self.elements[element.num] = element

    def add_set(self, name, elements, s_type):
        """Add an element set.

        Parameters
        ----------
        name : TODO

        Returns
        -------
        list :
            List containing the elements of the created set.

        """
        if s_type == "node":
            self.node_sets[name] = elements
            return self.node_sets[name]
        elif s_type == "element":
            self.element_sets[name] = elements
            return self.element_sets[name]

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

    def add_contact_pair(self, master, slave):
        """Add a contact pair to the model.

        Parameters
        ----------
        master : str
            Name of the surface defining the master surface.
        slave : str
            Name of the surface defining the slave surface.

        """
        self.contact_pairs += [{"master": master, "slave": slave}]

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

    def add_elem_output(self, elem, var, data, step, inc, intpnt):
        """Add element output data

        Parameters
        ----------
        var : TODO
        data : TODO
        intpnt : int
            Integration point number if the results contain integration point data.
            TODO: handle elements with different outputs.

        Returns
        -------
        TODO

        """
        if var not in self.elem_output[step][inc]:
            self.elem_output[step][inc][var] = dict()

        if elem not in self.elem_output[step][inc][var]:
            etype = self.elements[elem].elem_code
            self.elem_output[step][inc][var][elem] = np.empty((N_INT_PNTS[etype], 1),
                                                              dtype=float)

        self.elem_output[step][inc][var][elem][intpnt - 1] = data

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
            self.local_csys[n] = {inc_n: dict()}

        # Add increment to step
        else:
            step_time = data["step time"]
            load_prop = data["load proportionality"]
            time_inc = data["time increment"]
            inc_n = data["increment number"]

            # Initialize output repository for the current increment in step
            self.nodal_output[n][inc_n] = dict()
            self.elem_output[n][inc_n] = dict()
            self.local_csys[n][inc_n] = dict()

            self._curr_out_step = data["step number"]
            self._curr_incr = data["increment number"]

            self.steps[n].add_increment(inc_n, time_inc, step_time, load_prop)

    def add_local_csys(self, elem, csys, step, inc):
        """Add a local coordinate system asociated to the output of an element.

        Parameters
        ----------
        elem : int
            Identification number of the element to which the local coordinate
            system belongs.
        csys : arraylike
            Coordinate system as a 3x3 or 2x2 array, depending on the dimensionality of
            the model.

        """
        self.local_csys[step][inc][elem] = csys

    def get_nodal_result(self, var, step, inc, node_set=None, elem_set=None, node_ids=None):
        """Get nodal results

        Parameters
        ----------
        var : str
            Output variable
        step : int
            Number of the Abaqus step.
        inc : int
            Number of the increment.
        node_set : str, list
        elem_set : str, list

        Returns
        -------
        TODO

        """
        # Get the keys of the nodes in the set of nodes
        if node_set is not None:
            keys = sorted(self.get_nodes_from_set(node_set))
            elem_ids = self.get_elems_from_nodes(keys)
        # Get elements belonging to the set
        elif elem_set is not None:
            elem_ids = self.get_elems_from_set(elem_set)
            keys = sorted(self.get_nodes_from_elems(elem_ids))
        elif node_ids is not None:
            elem_ids = self.get_elems_from_nodes(node_ids)
            keys = sorted(node_ids)
        else:
            # FIXME: have this variable sorted globally
            keys = sorted(list(self.nodes.keys()))
            try:
                elem_ids = self.elem_output[step][inc][var].keys()
            except KeyError:
                print(f"Requested output variable {var} not present as element result of the model.")

        if var in self.nodal_output[step][inc]:
            results = self.nodal_output[step][inc][var]
        elif var in self.elem_output[step][inc]:
            results = self._nodal_result_from_elements(var, step, inc, elem_ids)
        else:
            # FIXME: handle errors properly some day
            print("Variable not present")

        list_res = [results[k] for k in keys]

        return np.asarray(list_res)

    def _nodal_result_from_elements(self, var, step, inc, elem_ids):
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
        keys_out = elem_ids
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
            res_nodes[elem_nodes] += nodal_i.flatten()
            counter[elem_nodes] += 1

        # FIXME: Another hacky fix
        counter[counter == 0] = np.nan

        result = res_nodes / counter

        # FIXME: output correct size
        return result

    def get_local_csys(self, direction, step, inc, elem_set=None):
        """Get the local coordinate system of each element (if present).

        Parameters
        ----------
        direction : str
            Direction of the local coordinate system.
        step : int
            Step number.
        inc : int
            Increment number within the step.
        elem_set : str, list, optional
            Element set.

        Returns
        -------
        arraylike :
            Array (3xn) or (2xn) depending on the dimensionality of the model.

        """
        if direction == "x":
            dim = 0
        elif direction == "y":
            dim = 1
        else:
            dim = 2

        # FIXME: have this variable sorted globally
        keys = sorted(list(self.elements.keys()))

        # Elements for which the output variable exists
        keys_out = set(self.local_csys[step][inc].keys())

        # Consider "deleted" elements (i.e. elements that have a 100% of damage according
        # to the used fracture model)
        if self._status is not None:
            status = self.elem_output[step][inc][f"SDV{self._status}"]
            del_elem = [k for k, v in status.items() if v[0] == 0]
            keys_out = [k for k in keys_out if k not in del_elem]
            keys = [k for k in keys if k not in del_elem]

        if elem_set is not None:
            keys_out, keys = self._map_elem_set_ids_to_output(elem_set, keys, keys_out)

        csys = self.local_csys[step][inc]

        nan_vec = np.ones(self._dimension) * np.nan

        list_res = [csys[k][dim, :] if k in keys_out else nan_vec for k in keys]

        return np.vstack(list_res)

    def _map_elem_set_ids_to_output(self, elem_set, id_elem, id_elem_out):
        """Get the keys of the elements in the element set mapped to the output array.

        Parameters
        ----------
        elem_set : TODO
        id_elem : TODO
        id_elem_out : TODO

        Returns
        -------
        keys_out : list
        keys : list

        """
        set_elements = self.get_elems_from_set(elem_set)

        def filter_elements(elem):
            if elem in set_elements:
                return True
            else:
                return False

        keys_out = filter(filter_elements, id_elem)
        keys = filter(filter_elements, id_elem_out)

        return keys_out, keys

    def get_time_history_result_from_node(self, var, node_id, steps="all"):
        """Get results for a node duiring the whole simulation.

        Parameters
        ----------
        var : TODO
        node_id : TODO
        steps : str, list, int
            The steps used to retrieve the results. Default: 'all'

        Returns
        -------
        np.asarray :
            Results for the given variable `var`

        """
        steps = self.steps.keys()

        # The first step is always zero (FIXME: maybe not always if there are
        # prestresses.)
        result = [0]
        for step in steps:
            for inc, val in self.nodal_output[step].items():
                result += [val[var][node_id]]

        return np.asarray(result)

    def get_nodal_vector_result(self, var, step, inc, node_set=None, elem_set=None):
        """Get the vector of a variable at each node.

        Parameters
        ----------
        var : str
            Output variable
        step : int
            Number of the Abaqus step.
        inc : int
            Number of the increment.
        node_set : str, list
        elem_set : str, list

        Returns
        -------
        array :
            Nx3 array of displacements in each node

        """
        coords = list()

        # Get the keys of the nodes in the set of nodes
        if node_set is not None:
            keys = sorted(self.get_nodes_from_set(node_set))
        # Get elements belonging to the set
        elif elem_set is not None:
            elem_ids = self.get_elems_from_set(elem_set)
            keys = sorted(self.get_nodes_from_elems(elem_ids))
        else:
            nodes = self.nodes
            keys = sorted(list(self.nodes.keys()))

        for k in keys:
            coords.append(self._get_node_vector_result(k, var, step, inc))

        coords_ar = np.asarray(coords)

        return coords_ar

    def get_element_result(self, var, step, inc, elem_set=None, elem_id=None):
        """Get element results.

        Parameters
        ----------
        var : str
            Variable that should be retrieved.
        step : int
            Number of the step.
        inc : int
            Increment number within the step.
        elem_set : str, list
            Element set from which the results should be retrieved.
        elem_id : int
            Number of a single element from which the results should be retrieved.

        Returns
        -------
        array :
            Results

        """
        # FIXME: have this variable sorted globally
        keys = sorted(list(self.elements.keys()))

        # Elements for which the output variable exists
        keys_out = set(self.elem_output[step][inc][var].keys())

        if self._status is not None:
            status = self.elem_output[step][inc][f"SDV{self._status}"]
            del_elem = [k for k, v in status.items() if v[0] == 0]
            keys_out = [k for k in keys_out if k not in del_elem]
            keys = [k for k in keys if k not in del_elem]

        if elem_set is not None:
            keys_out, keys = self._map_elem_set_ids_to_output(elem_set, keys, keys_out)

        elif elem_id is not None:
            set_elements = set(elem_id)

            def filter_elements(elem):
                if elem in set_elements:
                    return True
                else:
                    return False

            keys_out = set(elem_id)
            keys = elem_id

        results = self.elem_output[step][inc][var]

        list_res = [np.mean(results[k]) if k in keys_out else np.nan for k in keys]

        ar_results = np.asarray(list_res)

        return ar_results

    def get_surface_result(self, var, step, inc, surf_name):
        """Get element result on a given surface.

        Parameters
        ----------
        var : str
            Output variable.
        step : int
            Simulation step.
        inc : int
            Increment within the step.
        surface : str
            Name of the surface.

        Returns
        -------
        TODO

        """
        # Get underlying element numbers
        surf = self.surfaces[surf_name]
        e_nums = [face._element.num for face in surf._faces]

        # Retrieve element output
        out = self.get_element_result(var, step, inc, elem_id=e_nums)

        return out

    def add_metadata(self, metadata):
        """Add metadata to the model."""
        self.metadata[metadata[0]] = metadata[1]

    def get_node_coords(self, node_set=None, elem_set=None, node_id=None, return_map=False):
        """Get a list with the node coordinates.


        Parameters
        ----------
        node_set : str
            node_set
        elem_set : str, list
            elem_set
        node_id : int
            node_id
        return_map : bool
            return_map

        Returns
        -------
        coords : array
            An array of size (n, 3), where n is the number of nodes
        kmap : dict
            If either `node_set`, `elem_set` or `node_id` are given and `return_map` is
            `True`, then a dictionary is returned mapping the new node ids to the
            original node ids.

        """
        nodes = self.nodes

        if node_set is not None:
            old_keys = sorted(self.get_nodes_from_set(node_set))
            keys = np.arange(1, len(old_keys) + 1, 1)
            # Map new indices to old indices
            kmap = {k: ix for k, ix in zip(keys, old_keys)}
        elif node_id is not None:
            old_keys = sorted([node_id])
            keys = np.arange(1, len(old_keys) + 1, 1)
            # Map new indices to old indices
            kmap = {k: ix for k, ix in zip(keys, old_keys)}
        elif elem_set is not None:
            elems = self.get_elems_from_set(elem_set)
            old_keys = sorted(self.get_nodes_from_elems(elems))
            keys = np.arange(1, len(old_keys) + 1, 1)
            # Map new indices to old indices
            kmap = {k: ix for k, ix in zip(keys, old_keys)}
        else:
            #keys = sorted(list(nodes.keys()))
            #kmap = {k: k for k in keys}
            #the previous assumes that nodes are from 1 to n without jumps!!
            old_keys = sorted(list(nodes.keys()))
            keys = np.arange(1, len(old_keys) + 1, 1)
            # Map new indices to old indices
            kmap = {k: ix for k, ix in zip(keys, old_keys)}

        coords = np.empty((len(keys), 3))

        for k in keys:
            coords[k - 1, :] = nodes[kmap[k]].coords

        coords_ar = np.asarray(coords)

        if return_map:
            return coords_ar, kmap
        else:
            return coords_ar

    def get_deformed_node_coords(self, step, inc, scale=1, node_id=None, node_set=None,
                                 elem_set=None):
        """Get deformed node coordinates.

        Parameters
        ----------
        step : int
            Step to get deformations from
        inc : int
            Index of the increment in the required step.
        scale : float
            Multiply the deformations by this number.
        node_set : str, list
        elem_set : str, list

        Returns
        -------
        array :
            2D-Array with the node coordinates

        """
        coords, kmap = self.get_node_coords(node_id=node_id, node_set=node_set,
                                            elem_set=elem_set, return_map=True)

        for k in range(1, np.shape(coords)[0] + 1, 1):
            coords[k - 1, :] += self._get_node_vector_result(kmap[k], "U", step, inc) * scale

        return coords

    def get_cells(self, elem_set=None, status=None):
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

        # Element deletion is considered here
        if status is not None:

            def is_del(n_ele):
                if n_ele in status.keys():
                    if status[n_ele][0] != 0:
                        return True
                    else:
                        return False
                else:
                    return True

            # Don't consider the deleted elements for mesh
            elements = {k: v for k, v in elements.items() if is_del(k)}

        if elem_set is not None:
            elem_ids = self.get_elems_from_set(elem_set)
            nodes = self.get_nodes_from_elems(elem_ids)
            new_node_ids = np.arange(1, len(nodes) + 1, 1)
            kmap = {k: ix for k, ix in zip(nodes, new_node_ids)}
            elements = {k: elements[k] for k in elem_ids}
        else:
            node_ids = sorted(list(self.nodes.keys()))
            new_node_ids = np.arange(1, len(node_ids) + 1, 1)
            # Map new indices to old indices. kmap must be according with get_node_coords
            kmap = {ix: k for k, ix in zip(new_node_ids, node_ids)}

        keys = sorted(list(elements.keys()))

        cells = list()
        elem_type = list()

        for el_i in keys:
            cells.extend(elements[el_i].get_cell(kmap=kmap))
            elem_type.append(elements[el_i]._elem_type)

        ar_cells = np.asarray(cells)
        ar_elem_type = np.asarray(elem_type, np.int8)

        return ar_cells, ar_elem_type

    def get_mesh(self, elem_set=None):
        """Construct the mesh of the finite element model

        Parameters
        ----------
        elem_set : str
            Set of elements used to generate the mesh. If none is given, all elements
            are used.

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        nodes = self.get_node_coords(elem_set=elem_set)
        cells, elem_t = self.get_cells(elem_set)

        mesh = UnstructuredGrid(cells, elem_t, nodes)

        self.mesh = mesh

        return self.mesh

    def get_surface(self, name, return_nodes=False, step=None, inc=None, scale=1):
        """Get mesh of surface.

        Parameters
        ----------
        name : str
            Name of the surface
        return_nodes : bool (optional)
            Whether nodes should be returned separately as a list.
        step : int
            Step from which the deformation should be taken.
        inc : int
            Increment from which the deformation should be taken.
        scale : float
            Scaling factor of the deformation.

        Returns
        -------
        mesh : :obj:`UnstructuredGrid`
            Mesh representation of the surface.
        nodes : list
            Nodes corresponding to the mesh.

        """
        surface = self.surfaces[name]

        if return_nodes:
            return surface.mesh, surface.get_used_nodes().keys()
        else:
            return surface.mesh

    def get_deformed_mesh(self, step, inc, scale=1, elem_set=None):
        """Construct the deformed mesh in step with scaled deformations.

        Parameters
        ----------
        step : int
            Index of the needed step
        inc : int
            Index of the increment within the step
        scale : float
            Scale to be applied to the deformations
        status : int, None
            Solution-dependent state variable that controls the element deletion

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        nodes = self.get_deformed_node_coords(step, inc, scale, elem_set=elem_set)

        if self._status:
            status = self.elem_output[step][inc][f"SDV{self._status}"]
        else:
            status = None

        cells, elem_t = self.get_cells(elem_set=elem_set, status=status)

        mesh = UnstructuredGrid(cells, elem_t, nodes)

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
            u = np.asarray([
                nodal_output[f"{var}1"][n],
                nodal_output[f"{var}2"][n],
                nodal_output[f"{var}3"][n],
            ])
        else:
            u = np.asarray([
                nodal_output[f"{var}1"][n],
                nodal_output[f"{var}2"][n],
                0,
            ])

        return u

    def post_import_actions(self):
        """Execute some functions after importing all the records into the model."""
        pass

    def get_elems_from_set(self, elem_set):
        """Get the element IDs belonging to an element set.

        Parameters
        ----------
        elem_set : str, list
            Name of the set or list with names of different sets.

        Returns
        -------
        list :
            List containing the element IDs present in the set(s).

        """
        if isinstance(elem_set, str):
            elem_ids = self.element_sets[elem_set]
        # Is list
        else:
            elem_ids = []
            for set_i in elem_set:
                elem_ids += self.element_sets[set_i]

        return set(elem_ids)

    def get_nodes_from_elems(self, elems):
        """Get nodal IDs from a list of element IDs.

        Parameters
        ----------
        elems : list

        Returns
        -------
        TODO

        """
        elements = self.elements

        # Initialize list to store all the nodes
        nodes = list()

        for el in elems:
            nodes += elements[el]._nodes

        # Remove duplicates
        nodes_ar = np.asarray(nodes, dtype=int)

        return np.unique(nodes_ar)

    def get_nodes_from_set(self, node_set):
        """Get node IDs belonging to the node set.

        Parameters
        ----------
        node_set : str, list

        Returns
        -------
        TODO

        """
        if isinstance(node_set, str):
            node_ids = self.node_sets[node_set]
        # Is list
        else:
            node_ids = []
            for set_i in node_set:
                node_ids += self.node_sets[set_i]

        return node_ids

    def get_elems_from_nodes(self, node_ids):
        """Get element IDs from a set of nodes.

        Parameters
        ----------
        node_ids : list

        Returns
        -------
        TODO

        """
        nodes = self.nodes
        elem_ids = list()

        for ni in node_ids:
            elem_ids += nodes[ni].in_elements

        # Remove duplicates
        elems_ar = np.asarray(elem_ids, dtype=int)

        return np.unique(elems_ar)

    def __repr__(self):
        n_out = list(self.nodal_output[1][1].keys())
        e_out = list(self.elem_output[1][1].keys())
        s = f"""Abaqus result object:
--------------------
Number of nodes:          {len(self.nodes):,}
Number of elements:       {len(self.elements):,}
Number of node sets:      {len(self.node_sets):,}
Number of element sets:   {len(self.element_sets):,}
Nodal output variables:   {n_out}
Element output variables: {e_out}
"""
        return s
