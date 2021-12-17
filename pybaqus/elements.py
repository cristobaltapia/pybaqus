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
    def __init__(self, num, model, code):
        """TODO: to be defined.

        Parameters
        ----------
        id : TODO


        """
        self._num: int = num
        self._nodes: list = None
        self._n_nodes: int = None
        self._connectivity: list = None
        self._elem_code: str = code
        self._model = model
        self._elem_type = None
        self._faces: dict = dict()
        self._face_shape: dict = dict()
        self._n_integ_points: int = None
        self.map_node = np.vectorize(self._map_node)

    @property
    def elem_type(self):
        return self._elem_type

    @property
    def elem_code(self):
        return self._elem_code

    @property
    def model(self):
        return self._model

    @property
    def num(self):
        return self._num

    def get_nodes(self):
        return self._nodes

    def get_nodes_coords(self):
        """Get coordinates of nodes

        Returns
        -------
        TODO

        """
        all_nodes = self._model.nodes
        nodes = [all_nodes[n].coords for n in self._nodes]

        return nodes

    def get_cell(self, kmap=None):
        """Assemble cell array.

        Parameters
        ----------
        kmap : dict
            Map origianl node ids to new id

        """
        # Get nodes
        nodes = np.array(self._nodes)

        if kmap:
            nodes = self.map_node(kmap, nodes)

        nodes -= 1

        cell = np.array([self._n_nodes, *nodes], dtype=int)

        return cell

    def _map_node(self, kmap, old_id):
        """Map an old id to anew id.

        Parameters
        ----------
        old_id : TODO
        new_id : TODO

        Returns
        -------
        TODO

        """
        return kmap[old_id]

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

    @property
    def n_integ_points(self):
        return self._n_integ_points

    @n_integ_points.setter
    def n_integ_points(self, val):
        self._n_integ_points = val

    def extrapolate_to_nodes(self, s_int):
        """Extrapolate results computed on integration points to the nodes.

        Parameters
        ----------
        s_int : array
            Results on the integration points. (Order according to Abaqus definition.)

        Returns
        -------
        array

        """
        e_matrix = self._extrapol_matrix()
        s_n = e_matrix @ s_int
        return s_n, self._nodes

    def _extrapol_matrix(self):
        """Extrapolation matrix used to compute result variables at nodes."""

        e = self._elem_type
        print(f"Extrapolation matrix for element {e} needs to be defined.")

        return 0


class Quad(Element):
    """4-node rectangular element."""
    def __init__(self, n1, n2, n3, n4, num, model, code):
        super().__init__(num, model, code)
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

    def _extrapol_matrix(self):
        """Extrapolation matrix.

        Returns
        -------
        TODO

        """
        if self._n_integ_points == 4:
            ext_mat = np.array([
                [1.8660254, -0.5, 0.1339746, -0.5],
                [-0.5, 1.8660254, -0.5, 0.1339746],
                [0.1339746, -0.5, 1.8660254, -0.5],
                [-0.5, 0.1339746, -0.5, 1.8660254],
            ])
        # One integration point
        else:
            ext_mat = np.ones((4, 1))

        return ext_mat

    def N1(self, xi, eta):
        """Shape function for node 1."""
        n1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        return n1

    def N2(self, xi, eta):
        """Shape function for node 2."""
        n2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        return n2

    def N3(self, xi, eta):
        """Shape function for node 3."""
        n3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        return n3

    def N4(self, xi, eta):
        """Shape function for node 4."""
        n4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        return n4


class Triangle(Element):
    """3-node triangular element."""
    def __init__(self, n1, n2, n3, num, model, code):
        super().__init__(num, model, code)
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

    def _extrapol_matrix(self):
        """Extrapolation matrix.

        Returns
        -------
        TODO

        """
        if self._n_integ_points == 1:
            ext_mat = np.ones((3, 1))
        else:
            ext_mat = None

        return ext_mat


class Tetra(Element):
    """4 node tetrahedron elements"""
    def __init__(self, n1, n2, n3, n4, num, model, code):
        super().__init__(num, model, code)
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
    def __init__(self, n1, n2, n3, n4, n5, num, model, code):
        super().__init__(num, model, code)
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
    def __init__(self, n1, n2, n3, n4, n5, n6, num, model, code):
        super().__init__(num, model, code)
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

    def _extrapol_matrix(self):
        """Extrapolation matrix.

        Returns
        -------
        TODO

        """
        if self._n_integ_points == 2:
            ext_mat = np.array([
                [1.3660254037844386, -0.3660254037844386],
                [1.3660254037844386, -0.3660254037844386],
                [1.3660254037844386, -0.3660254037844386],
                [-0.3660254037844386, 1.3660254037844386],
                [-0.3660254037844386, 1.3660254037844386],
                [-0.3660254037844386, 1.3660254037844386],
            ])
        # One integration point
        else:
            ext_mat = np.ones((1, 1))

        return ext_mat


class Hexahedron(Element):
    """8 node brick element."""
    def __init__(self, n1, n2, n3, n4, n5, n6, n7, n8, num, model, code):
        super().__init__(num, model, code)
        self._n_nodes = 8
        self._nodes = [n1, n2, n3, n4, n5, n6, n7, n8]
        self._elem_type = vtk.VTK_HEXAHEDRON

        # Define faces connectivity
        self._faces = {
            1: [0, 1, 2, 3],
            2: [4, 7, 6, 5],
            3: [0, 4, 5, 1],
            4: [1, 5, 6, 2],
            5: [2, 6, 7, 3],
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

    def _extrapol_matrix(self):
        """Extrapolation matrix.

        Returns
        -------
        TODO

        """
        if self._n_integ_points == 8:
            ext_mat = np.array([
                [
                    2.549038105676658, -0.6830127018922192, 0.18301270189221927,
                    -0.6830127018922192, -0.6830127018922192, 0.18301270189221927,
                    -0.04903810567665795, 0.18301270189221927
                ],
                [
                    -0.6830127018922192, 2.549038105676658, -0.6830127018922192,
                    0.18301270189221927, 0.18301270189221927, -0.6830127018922192,
                    0.18301270189221927, -0.04903810567665795
                ],
                [
                    -0.6830127018922192, 0.18301270189221927, -0.6830127018922192,
                    2.549038105676658, 0.18301270189221927, -0.04903810567665795,
                    0.18301270189221927, -0.6830127018922192
                ],
                [
                    0.18301270189221927, -0.6830127018922192, 2.549038105676658,
                    -0.6830127018922192, -0.04903810567665795, 0.18301270189221927,
                    -0.6830127018922192, 0.18301270189221927
                ],
                [
                    -0.6830127018922192, 0.18301270189221927, -0.04903810567665795,
                    0.18301270189221927, 2.549038105676658, -0.6830127018922192,
                    0.18301270189221927, -0.6830127018922192
                ],
                [
                    0.18301270189221927, -0.6830127018922192, 0.18301270189221927,
                    -0.04903810567665795, -0.6830127018922192, 2.549038105676658,
                    -0.6830127018922192, 0.18301270189221927
                ],
                [
                    0.18301270189221927, -0.04903810567665795, 0.18301270189221927,
                    -0.6830127018922192, -0.6830127018922192, 0.18301270189221927,
                    -0.6830127018922192, 2.549038105676658
                ],
                [
                    -0.04903810567665795, 0.18301270189221927, -0.6830127018922192,
                    0.18301270189221927, 0.18301270189221927, -0.6830127018922192,
                    2.549038105676658, -0.6830127018922192
                ],
            ])
        # One integration point
        else:
            ext_mat = np.ones((8, 1))

        return ext_mat


class LineElement(Element):
    """2 node line element."""
    def __init__(self, n1, n2, num, model, code):
        super().__init__(num, model, code)
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


class QuadraticHexahedron(Element):
    """20 node quadratic brick element."""
    def __init__(self, n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15,
                 n16, n17, n18, n19, n20, num, model, code):
        super().__init__(num, model, code)
        self._n_nodes = 20
        self._nodes = [
            n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18,
            n19, n20
        ]
        self._elem_type = vtk.VTK_QUADRATIC_HEXAHEDRON

        # Define faces connectivity
        self._faces = {
            1: [0, 1, 2, 3, 8, 9, 10, 11],
            2: [4, 7, 6, 5, 15, 14, 13, 12],
            3: [0, 4, 5, 1, 16, 12, 17, 8],
            4: [1, 5, 6, 2, 17, 13, 18, 9],
            5: [2, 6, 7, 3, 18, 14, 19, 10],
            6: [3, 7, 4, 0, 19, 15, 16, 11],
        }
        self._face_shape = {
            1: vtk.VTK_QUADRATIC_QUAD,
            2: vtk.VTK_QUADRATIC_QUAD,
            3: vtk.VTK_QUADRATIC_QUAD,
            4: vtk.VTK_QUADRATIC_QUAD,
            5: vtk.VTK_QUADRATIC_QUAD,
            6: vtk.VTK_QUADRATIC_QUAD,
        }


ELEMENTS = {
    # Rigid
    "R2D2": LineElement,
    "R3D3": Triangle,
    "R3D4": Quad,
    # 2D Continuum
    "S4R": Quad,
    "CAX4": Quad,
    "CPS4": Quad,
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
    "C3D20R": QuadraticHexahedron,
}

N_INT_PNTS = {
    # Rigid
    "R2D2": 0,
    "R3D3": 0,
    "R3D4": 0,
    # 2D Continuum
    "S4R": 1,
    "CAX4": 4,
    "CPS4": 4,
    "CPS4R": 1,
    "CPE3": 1,
    "CPE3H": 1,
    "CPS3": 1,
    "CPEG3": 1,
    # 3D Continuum
    "C3D4": 1,
    "C3D4H": 1,
    "C3D6": 2,
    "C3D6H": 2,
    "C3D8": 8,
    "C3D8H": 1,
    "C3D8R": 1,
    "C3D8RS": 1,
    "C3D20R": 8,
}
