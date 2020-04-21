"""
Definitions for the different element types.
"""
import numpy as np
import vtk


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
        self._elem_type = None
        self._faces: dict = dict()
        self._face_shape: dict = dict()

    @property
    def elem_type(self):
        """TODO: Docstring for elem_type.
        Returns
        -------
        TODO

        """
        return self._elem_type

    def get_nodes(self):
        return self._nodes

    def get_nodes_coords(self):
        """Get coordinates of nodes

        Returns
        -------
        TODO

        """
        all_nodes = self.model.nodes
        nodes = [all_nodes[n].coords for n in self._nodes]

        return nodes

    def get_cell(self):
        # Get nodes
        nodes = np.array(self._nodes) - 1

        cell = np.array([self._n_nodes, *nodes], dtype=int)

        return cell

    def get_face(self, face):
        """Get a specific face of the element.

        Parameters
        ----------
        face : TODO

        Returns
        -------
        TODO

        """
        face = self._faces[face]

        return face

    def get_face_shape(self, face):
        """TODO: Docstring for get_face_shape.

        Parameters
        ----------
        face : int
            Number of the face.

        Returns
        -------
        vtk_shape :
            The correct shape of the face.

        """
        return self._face_shape[face]


class Quad(Element):
    """4-node rectangular element."""

    def __init__(self, n1, n2, n3, n4, num, model):
        super().__init__(num, model)
        self._n_nodes = 4
        self._nodes = [n1, n2, n3, n4]
        self._elem_type = vtk.VTK_QUAD

        # Define faces connectivity
        self._faces = {
            1: [0, 1],
            2: [1, 2],
            3: [2, 3],
            4: [3, 0],
            7: [0, 1, 2, 3],
            8: [3, 2, 1, 0],
        }
        self._face_shape = {
            1: vtk.VTK_LINE,
            2: vtk.VTK_LINE,
            3: vtk.VTK_LINE,
            4: vtk.VTK_LINE,
            7: vtk.VTK_QUAD,
            8: vtk.VTK_QUAD,
        }


class Triangle(Element):
    """3-node triangular element."""

    def __init__(self, n1, n2, n3, num, model):
        super().__init__(num, model)
        self._n_nodes = 3
        self._nodes = [n1, n2, n3]
        self._elem_type = vtk.VTK_TRIANGLE

        # Define faces connectivity
        self._faces = {1: [0, 1], 2: [1, 2], 3: [2, 0], 7: [0, 1, 2], 8: [2, 1, 0]}
        self._face_shape = {
            1: vtk.VTK_LINE,
            2: vtk.VTK_LINE,
            3: vtk.VTK_LINE,
            7: vtk.VTK_TRIANGLE,
            8: vtk.VTK_TRIANGLE,
        }


class Tetra(Element):
    """4 node tetrahedron elements"""

    def __init__(self, n1, n2, n3, n4, num, model):
        super().__init__(num, model)
        self._n_nodes = 4
        self._nodes = [n1, n2, n3, n4]
        self._elem_type = vtk.VTK_TETRA

        # Define faces connectivity
        self._faces = {1: [0, 1, 2], 2: [0, 4, 2], 3: [1, 3, 2], 4: [2, 3, 0]}
        self._face_shape = {
            1: vtk.VTK_TRIANGLE,
            2: vtk.VTK_TRIANGLE,
            3: vtk.VTK_TRIANGLE,
        }


class Pyramid(Element):
    """5 node pyramid element."""

    def __init__(self, n1, n2, n3, n4, n5, num, model):
        super().__init__(num, model)
        self._n_nodes = 5
        self._nodes = [n1, n2, n3, n4, n5]
        self._elem_type = vtk.VTK_PYRAMID

        # Define faces connectivity
        self._faces = {
            1: [0, 1, 2, 3],
            2: [0, 4, 1],
            3: [1, 4, 2],
            4: [2, 4, 3],
            5: [3, 4, 0],
        }
        self._face_shape = {
            1: vtk.VTK_QUAD,
            2: vtk.VTK_TRIANGLE,
            3: vtk.VTK_TRIANGLE,
            4: vtk.VTK_TRIANGLE,
        }


class Wedge(Element):
    """6 node triangular prism element."""

    def __init__(self, n1, n2, n3, n4, n5, n6, num, model):
        super().__init__(num, model)
        self._n_nodes = 6
        self._nodes = [n1, n2, n3, n4, n5, n6]
        self._elem_type = vtk.VTK_WEDGE

        # Define faces connectivity
        self._faces = {
            1: [0, 1, 2],
            2: [3, 5, 4],
            3: [0, 3, 4, 1],
            4: [1, 4, 5, 2],
            5: [2, 5, 3, 0],
        }
        self._face_shape = {
            1: vtk.VTK_TRIANGLE,
            2: vtk.VTK_TRIANGLE,
            3: vtk.VTK_QUAD,
            4: vtk.VTK_QUAD,
            5: vtk.VTK_QUAD,
        }


class Hexahedron(Element):
    """8 node brick element."""

    def __init__(self, n1, n2, n3, n4, n5, n6, n7, n8, num, model):
        super().__init__(num, model)
        self._n_nodes = 8
        self._nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
        self._elem_type = vtk.VTK_PYRAMID

        # Define faces connectivity
        self._faces = {
            1: [0, 1, 2, 3],
            2: [4, 7, 6, 5],
            3: [0, 4, 5, 1],
            4: [1, 5, 6, 1],
            5: [2, 6, 7, 4],
            6: [3, 7, 4, 0],
        }
        self._face_shape = {
            1: vtk.VTK_QUAD,
            2: vtk.VTK_QUAD,
            3: vtk.VTK_QUAD,
            4: vtk.VTK_QUAD,
            5: vtk.VTK_QUAD,
            6: vtk.VTK_QUAD,
        }


class LineElement(Element):
    """2 node line element."""

    def __init__(self, n1, n2, num, model):
        super().__init__(num, model)
        self._n_nodes = 2
        self._nodes = [n1, n2]
        self._elem_type = vtk.VTK_LINE

        # Define faces connectivity
        self._faces = {1: [0, 1], 7: [0, 1], 8: [1, 0]}
        self._face_shape = {
            1: vtk.VTK_LINE,
            7: vtk.VTK_LINE,
            8: vtk.VTK_LINE,
        }


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

