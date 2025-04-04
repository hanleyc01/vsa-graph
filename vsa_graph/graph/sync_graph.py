from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from vsa_graph.vsa.hrr import HRR


@dataclass
class Graph:
    """A `Graph` is a list of connections.

    Call `self.run` in order to run the connections. Note that operations
    are run *in the order* that they are given as `connections`. This means
    that we could potentially have operations run out-of-order.
    """

    connections: t.List[Connection]

    def run(self) -> None:
        for connection in self.connections:
            connection.forward()


type Label = str
"""Type alias for str."""

type ConnectionItem = t.Tuple[Label, Node]
"""A `ConnectionItem` contains label name for the output buffer of the 
incoming node to a connection, and the Node itself.
"""


@dataclass
class Connection:
    """A `Connection` is an edge in our computational graph.

    Args:
        input_nodes (list[tuple[str, Node]]): A list of output buffer labels
            and nodes. Note that the order of tuples is preserved when used
            as arguments to `self.output_node.__call__`.
        output_node (Node): The the output vertex of the edge.
            `self.input_nodes` are applied as arguments to
            `self.output_node.__call__` in order.
    """

    input_nodes: t.List[ConnectionItem]
    output_node: Node

    def forward(self) -> None:
        """Feed the buffers specified in `self.input_nodes` to `self.output_node.__call__`.
        Note that order is preserved.
        """
        print("Graph forward!")
        input_args = []
        for output_buffer_label, node in self.input_nodes:
            print(f"Getting {output_buffer_label} from {node.label}")
            input_args.append(node[output_buffer_label])

        print(f"Running {self.output_node.label} with args!")
        self.output_node(input_args)


class Node(ABC):
    """Abstract base class of Nodes in the computational graph.

    Each subclass of Node must have the following class members and methods,
    though does not need to implement them.

    Warning: labels of buffers should not have conflicting names! This will
    mess up feeding forwards.
    """

    label: Label
    """
    The name of the Node.
    """
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    """
    Named input buffers, which hold immutable references to buffers from
    other nodes.
    """
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    """The output buffer which is the result of `__call__`.
    """

    @abstractmethod
    def __call__(self, inputs: t.List[npt.NDArray[t.Any]]) -> None:
        """Use `inputs` in some operation, putting the result in
        `self.output_buffers["out"]`.
        """
        ...

    @abstractmethod
    def __getitem__(self, key: str) -> npt.NDArray[t.Any]:
        """Retrieve a buffer by label `key`."""
        ...


class Bind(Node):
    """A HRR binding node.

    HRR binding is implemented through circular convolution of the input
    buffers, outputting the result in `self.output_buffer["out"]`.
    """

    label: Label
    dim: int
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]

    def __init__(self, label: Label, dim: int) -> None:
        self.label = label
        self.dim = dim

        self.input_buffers = {"lhs": np.zeros(dim), "rhs": np.zeros(dim)}
        self.output_buffers = {"out": np.zeros(dim)}

    def __getitem__(self, key: Label) -> npt.NDArray[np.float64]:
        input_buff = self.input_buffers.get(key)
        if input_buff is None:
            return self.output_buffers[key]
        else:
            return input_buff

    def __call__(self, inputs: t.List[npt.NDArray[np.float64]]) -> None:
        assert len(inputs) == 2, "nargs of bind == 2"
        for item in inputs:
            assert (
                self.dim == item.size
            ), f"expected dim: {self.dim} got {item.size}"

        self.input_buffers["lhs"] = inputs[0]
        self.input_buffers["rhs"] = inputs[1]

        self.output_buffers["out"] = HRR.bind(
            self.input_buffers["lhs"], self.input_buffers["rhs"]
        )

        self.__post_call__()

    def __post_call__(self) -> None:
        for key in self.input_buffers.keys():
            self.input_buffers[key] = np.zeros(self.dim)


class UserInput(Node):
    """User input computational graph node.

    Used for prespecifying inputs to a computational graph.
    """

    label: Label
    buffer: t.Dict[Label, npt.NDArray[t.Any]]
    dim: int

    def __init__(self, label: str, input: npt.NDArray[t.Any]) -> None:
        self.label = label
        self.buffer = {"out": input}
        self.dim = input.size

    def __getitem__(self, key: Label) -> npt.NDArray[np.float64]:
        input_buff = self.input_buffers.get(key)
        if input_buff is None:
            return self.output_buffers[key]
        else:
            return input_buff

    def __call__(self, items: t.List[npt.NDArray[t.Any]]) -> None:
        raise NotImplementedError("bad")
