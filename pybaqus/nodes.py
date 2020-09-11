"""
Definition of node objects.
"""
import numpy as np


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
        self._in_elements = None

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z

    @property
    def rx(self):
        return self._rx

    @property
    def ry(self):
        return self._ry

    @property
    def rz(self):
        return self._rz

    @property
    def coords(self):
        return self._get_coords()

    @property
    def in_elements(self):
        return self._in_elements

    @in_elements.setter
    def in_elements(self, x):
        self._in_elements = x

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
