from __future__ import annotations
import typing as t
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vsa.hrr import HRR

# optimization here is to enforce the computational graph as a DAG;
# if it is a dag, we can run branches asynchronously, and then block on join
# points.
#
# The rough idea is something like this
#
# Run async           Async again
# ┌─┬────────►┌─────┬────────────►┌─┬───────►┌─┐
# └─┘         │     │             └─┘        └─┘
# ┌─┬────────►│     │                        ▲
# └─┘         │     ├────────────►┌─┐        │
# ┌─┬────────►│     │             └─┴────────┘
# └─┘         └─────┴────────────►┌─┬───────►┌─┐
#            Block on             └─┘        └─┘
#
# In human terms: we'd run branches asynchronously and await their output
# at joining nodes or if they are a final output node.
#
# We might also want to think about directed graphs *in general*, as they
# would allow for complete computational specification.

# One way that we could go about doing this is to organize the graph by depth.
# This would, of course, put the onus on the user to be safe, but the idea here
# is that each depth of the graph would be connections which can be
# safely run in parallel.


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
        print("Graph forward!")
        input_args = []
        for output_buffer_label, node in self.input_nodes:
            print(f"Getting {output_buffer_label} from {node.label}")
            input_args.append(node[output_buffer_label])

        print(f"Running {self.output_node.label} with args!")
        self.output_node(input_args)


class Node(ABC):
    label: Label

    @abstractmethod
    def __call__(self, inputs: t.List[npt.NDArray[t.Any]]) -> None: ...

    @abstractmethod
    def __getitem__(self, key: str) -> npt.NDArray[t.Any]: ...


type LabelBuffer = t.Tuple[Label, npt.NDArray[np.float64]]

_label_count = -1


def _fresh_label(x: str) -> Label:
    global _label_count
    _label_count += 1
    return f"{x}{_label_count}"


class Bind(Node):
    label: Label
    dim: int
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]

    def __init__(self, dim: int, label: Label | None = None) -> None:
        if label is None:
            self.label = _fresh_label("Bind")
        else:
            self.label = label
        self.dim = dim

        self.input_buffers = {"lhs": np.zeros(dim), "rhs": np.zeros(dim)}
        self.output_buffers = {"out": np.zeros(dim)}

    def __getitem__(self, key: Label) -> npt.NDArray[np.float64]:
        return self.output_buffers[key]

    def __call__(self, inputs: t.List[npt.NDArray[np.float64]]) -> None:
        assert len(inputs) == 2, "nargs of bind == 2"
        for item in inputs:
            assert (
                self.dim == item.size
            ), f"expected dim: {self.dim} got {item.size}"

        for i in range(inputs[0].size):
            self.input_buffers["lhs"][i] = inputs[0][i]

        for i in range(inputs[1].size):
            self.input_buffers["rhs"][i] = inputs[1][i]

        self.output_buffers["out"] = HRR.bind(
            self.input_buffers["lhs"], self.input_buffers["rhs"]
        )

        self.__post_call__()

    def __post_call__(self) -> None:
        for key in self.input_buffers.keys():
            self.input_buffers[key] = np.zeros(self.dim)


class UserInput(Node):
    label: Label
    buffer: t.Dict[Label, npt.NDArray[t.Any]]
    dim: int

    def __init__(self, label: str, input: npt.NDArray[t.Any]) -> None:
        self.label = label
        self.buffer = {"out": input}
        self.dim = input.size

    def __getitem__(self, key: str) -> npt.NDArray[t.Any]:
        return self.buffer[key]

    def __call__(self, items: t.List[npt.NDArray[t.Any]]) -> None:
        raise NotImplementedError("bad")
