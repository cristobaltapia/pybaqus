"""
Implementation of faces
"""
import numpy as np


class Surface:
    """Surface object.

    Parameters
    ----------
    name : str
        Name of the surface

    """

    def __init__(self, name, dimension):
        self._name = name
        self._faces: list = list()
        self._type: str = None
        dimension_map = {1: "1-D", 2: "2-D", 3: "3-D", 4: "Axysymmetric"}
        self._dimension = dimension_map[dimension]

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

    def __init__(self, name, dimension, ref_point):
        super().__init__(name, dimension)
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

    def __init__(self, name, dimension, master_surfaces):
        super().__init__(name, dimension)
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
