import pytest
from pybaqus.model import Model

@pytest.fixture
def empty_model():
    return Model()

def test_model_initialization(empty_model):
    assert isinstance(empty_model, Model)
    assert len(empty_model.nodes) == 0
    assert len(empty_model.elements) == 0
    assert len(empty_model.element_sets) == 0
    assert len(empty_model.node_sets) == 0
    assert len(empty_model.surfaces) == 0

def test_add_node(empty_model):
    class MockNode:
        def __init__(self, num):
            self._num = num

    node = MockNode(1)
    empty_model.add_node(node)
    assert len(empty_model.nodes) == 1
    assert empty_model.nodes[1] == node

def test_add_element(empty_model):
    class MockElement:
        def __init__(self, num):
            self.num = num

    element = MockElement(1)
    empty_model.add_element(element)
    assert len(empty_model.elements) == 1
    assert empty_model.elements[1] == element

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

