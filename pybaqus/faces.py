"""
Implementation of faces
"""
import numpy as np
from copy import copy

from pyvista import UnstructuredGrid


class Surface:
    """Surface object.

    Parameters
    ----------
    name : str
        Name of the surface

    """

    def __init__(self, name, dimension, model):
        self._name = name
        self._model = model
        self._faces: list = list()
        self._type: str = None
        dimension_map = {1: "1-D", 2: "2-D", 3: "3-D", 4: "Axysymmetric"}
        self._dimension = dimension_map[dimension]
        self._mesh = None

    @property
    def dimension(self):
        return self._dimension

    @property
    def mesh(self):
        """TODO: Docstring for mesh.
        Returns
        -------
        TODO

        """
        if self._mesh is not None:
            return copy(self._mesh)
        else:
            return copy(self._gen_mesh())

    def __str__(self):
        text = (
            f"{self._type} surface '{self._name}'\n"
            + f"Dimension: {self._dimension}\n"
            + f"Number of faces: {len(self._faces)}\n"
        )
        return text

    def add_face(self, face):
        """Add a face to the surface

        Parameters
        ----------
        face : `obj`:Face

        """
        self._faces.append(face)

    def get_cells(self):
        """Get face geometries of the surface.

        Parameters
        ----------
        step : TODO
        inc : TODO
        scale : TODO

        Returns
        -------
        TODO

        """
        faces = self._faces

        u_nodes = self.get_used_nodes()

        # Remap node keys
        sorted_nodes = sorted(u_nodes.keys())
        new_keys = {old: new for new, old in enumerate(sorted_nodes)}
        new_nodes = np.array([u_nodes[k].coords for k in sorted_nodes])

        list_nodes = [y for fi in faces for y in self._surf_data(fi, new_keys)]
        elem_type = [fi.element_type for fi in faces]

        ar_cells = np.array(list_nodes)
        ar_elem_type = np.array(elem_type, np.int8)

        return ar_cells, ar_elem_type, new_nodes

    def get_used_nodes(self):
        """Get the nodes belonging to the surface.

        Returns
        -------
        dict :
            Dictionary with the used nodes.

        """
        node_list = []
        for fi in self._faces:
            node_list += fi.get_nodes()

        # nodes = np.unique(np.array([fi.get_nodes() for fi in self._faces]))
        nodes = np.unique(np.array(node_list))

        all_nodes = self._model.nodes
        used_nodes = {k: all_nodes[k] for k in nodes}

        return used_nodes

    def _gen_mesh(self):
        """Construct the mesh of the finite element model

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        cells, elem_t, nodes = self.get_cells()

        mesh = UnstructuredGrid(cells, elem_t, nodes)

        self._mesh = mesh

        return self._mesh

    def _surf_data(self, face, new_keys):
        """Get nodes, offset and element type of face element.

        Parameters
        ----------
        face : TODO

        Returns
        -------
        TODO

        """
        global_nodes = face.get_nodes()
        local_nodes = [new_keys[k] for k in global_nodes]

        nodes = [face._n_nodes] + local_nodes
        return nodes


class RigidSurface(Surface):
    """Rigid surface object.

    Parameters
    ----------
    name : str
        Label of the surface.
    dimension : int
        Dimension of the surface.
    ref_point : int
        Label of the reference node for the rigid surface.

    """

    def __init__(self, name, dimension, model, ref_point):
        super().__init__(name, dimension, model)
        self._ref_point = ref_point
        self._type = "Rigid"


class DeformableSurface(Surface):
    """Deformable surface object.

    Parameters
    ----------
    name : str
        Label of the surface.
    dimension : int
        Dimension of the surface.
    master_surfaces : list
        List of the master surface names.

    """

    def __init__(self, name, dimension, model, master_surfaces):
        super().__init__(name, dimension, model)
        self._master_surfaces = master_surfaces
        self._type = "Deformable"


class Face:
    """Defines an element Face object.

    Parameters
    ----------
    element : int
        Index of the underlying element of the face.
    face : int
        Number of the internal face of the element.
    nodes : list
        List of node labels comprising the face.

    """

    def __init__(self, element, face, nodes):
        self._element = element
        self._face = face
        self._nodes = nodes
        self._n_nodes = len(nodes)

    def get_nodes(self):
        """Get the vertices of the face."""
        # return self._element.get_face(self._face)
        return self._nodes

    @property
    def element_type(self):
        """Return the element type of the surface"""
        return self._element.get_face_shape(self._face)
