import pytest

from pybaqus.model import Model
from pybaqus.nodes import Node2D, Node3D


@pytest.fixture
def empty_model():
    return Model()


# @pytest.fixture
# def new_2D_node():
#     return Node2D(1, dof_map=[])


def test_model_initialization(empty_model):
    assert isinstance(empty_model, Model)
    assert empty_model.nodes == None
    assert empty_model.elements == None
    assert len(empty_model.element_sets) == 0
    assert len(empty_model.node_sets) == 0
    assert len(empty_model.surfaces) == 0


def test_add_node(empty_model):
    class MockNode:
        def __init__(self, num):
            self._num = num

    node = MockNode(0)
    empty_model.size = (1, 1)
    empty_model.add_node(node)
    assert len(empty_model.nodes) == 1
    assert empty_model.nodes[0] == node


def test_add_element(empty_model):
    class MockElement:
        def __init__(self, num):
            self.num = num

    element = MockElement(0)
    empty_model.size = (1, 1)
    empty_model.add_element(element)
    assert len(empty_model.elements) == 1
    assert empty_model.elements[0] == element


def test_add_set(empty_model):
    elements = [1, 2, 3]
    result = empty_model.add_set("test_set", elements, "element")
    assert "test_set" in empty_model.element_sets
    assert empty_model.element_sets["test_set"] == elements
    assert result == elements

    nodes = [4, 5, 6]
    result = empty_model.add_set("test_node_set", nodes, "node")
    assert "test_node_set" in empty_model.node_sets
    assert empty_model.node_sets["test_node_set"] == nodes
    assert result == nodes
