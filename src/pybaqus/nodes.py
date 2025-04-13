"""
Definition of node objects.
"""

import numpy as np


class Node:
    """Define a node in the finite element model.

    Attributes
    ----------
    x : float
    y : float
    z : float
    num : float
    in_elements : list[int]
        List of elements that use this node.

    """

    _x: float
    _y: float
    _z: float
    _rx: float
    _ry: float
    _rz: float

    def __init__(self, num: int, model):
        self._num: int = num
        self.model = model
        self._in_elements = None

    @property
    def id(self):
        return self._num

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
    num : int
        ID of the node
    dof_map : dict
        Dictionary mapping the active DOF to each nodal output position
    model : `obj`:Model
        Model to which the node belongs to.
    *dof :
        The values for all degrees of freedom.

    """

    def __init__(self, num: int, dof_map, model, *dof):
        super().__init__(num, model)

        self._x = dof[dof_map[0] - 1]
        self._y = dof[dof_map[1] - 1]
        self._rz = dof[dof_map[5] - 1] if dof_map[5] > 0 else np.nan
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

        self._x = dof[dof_map[0] - 1]
        self._y = dof[dof_map[1] - 1]
        self._z = dof[dof_map[2] - 1]
        self._rx = dof[dof_map[3] - 1] if dof_map[3] > 0 else np.nan
        self._ry = dof[dof_map[4] - 1] if dof_map[4] > 0 else np.nan
        self._rz = dof[dof_map[5] - 1] if dof_map[5] > 0 else np.nan
        self._num = num

    def _get_coords(self):
        return (self._x, self._y, self._z)
