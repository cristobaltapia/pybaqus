"""
Implementation of faces
"""
import numpy as np
import vtk

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
            return self._mesh
        else:
            return self._gen_mesh()

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
        new_keys = {old: new for new, old in enumerate(u_nodes.keys())}
        new_nodes = np.array([u_nodes[k].coords for k in u_nodes.keys()])

        list_nodes = list()
        offset = list()
        elem_type = list()

        for face in faces:
            global_nodes = face.get_nodes()
            local_nodes = [new_keys[k] for k in global_nodes]

            nodes = [face._n_nodes] + local_nodes
            list_nodes.extend(nodes)
            offset.append(face._n_nodes + 1)
            elem_type.append(face.element_type)

        ar_cells = np.array(list_nodes)
        ar_offset = np.cumsum(np.array(offset, dtype=int)) - offset[0]
        ar_elem_type = np.array(elem_type, np.int8)

        return ar_cells, ar_offset, ar_elem_type, new_nodes

    def get_used_nodes(self):
        """Get the nodes belonging to the surface.

        Returns
        -------
        dict :
            Dictionary with the used nodes.

        """
        nodes = list()
        for face in self._faces:
            nodes.extend(face.get_nodes())

        # Remove duplicates
        nodes = list(set(nodes))

        all_nodes = self._model.nodes
        used_nodes = {k: n for k, n in all_nodes.items() if k in nodes}

        return used_nodes

    def _gen_mesh(self):
        """Construct the mesh of the finite element model

        Returns
        -------
        mesh : mesh
            VTK mesh unstructured grid

        """
        cells, offset, elem_t, nodes = self.get_cells()

        mesh = UnstructuredGrid(offset, cells, elem_t, nodes)

        self._mesh = mesh

        return self._mesh


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
