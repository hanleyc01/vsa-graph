from __future__ import annotations

import typing as t
from abc import ABC
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from vsa_graph.vsa.hrr import HRR


@dataclass
class Graph:
    """A Graph is a list of Connections, which allows for delaying and
    visualizing execution of vector-symbolic operations.
    """

    connections: t.List[Connection]

    def forward(self) -> None:
        """Feed each connection input into the edge function, and put
        the result into the output buffer.
        """
        for connection in self.connections:
            connection.edge(connection.inputs, connection.outputs)

    def display(self) -> None:
        """Display the graph as a list of nodes and edges."""
        print("Graph:")
        for connection in self.connections:
            print(
                f"\t{[input.label for input in connection.inputs]} --[{connection.edge.label}]---> {[output.label for output in connection.outputs]}"
            )


type ArrayF64 = npt.NDArray[np.float64]
"""Type alias for `npt.NDArray[np.float64]`."""


@dataclass
class Connection:
    """A graph edge, connecting input buffers to output buffers, with edge
    function `self.edge`.
    """

    edge: GraphEdge
    inputs: t.List[Vertex]
    outputs: t.List[Vertex]


class GraphEdge(ABC):
    """Abstract base class of edges in the computational graph.

    In order to implement a `GraphEdge`, the edge must implement
    `GraphEdge.__call__(self, t.List[Vertex], t.List[Vertex])`, with the
    first parameter being a list of input buffers and the second being a list
    of output buffers.
    """
    label: str

    def __call__(
        self, inputs: t.List[Vertex], outputs: t.List[Vertex]
    ) -> None: ...


@dataclass
class Bind(GraphEdge):
    _count: int = 0
    label: str = field(init=False)

    def __post_init__(self) -> None:
        Bind._count += 1
        self.label = f"Bind{Bind._count}"

    def __call__(
        self, inputs: t.List[Vertex], outputs: t.List[Vertex]
    ) -> None:
        nargs = len(inputs)
        nouts = len(outputs)
        if nargs != 2:
            raise ValueError(f"innapropriate number of inputs: {len(inputs)}")
        if nouts != 1:
            raise ValueError(
                f"innapropriate number of outputs: {len(outputs)}"
            )

        lhs, rhs = inputs[0].data, inputs[1].data
        outputs[0].data = HRR.bind(lhs, rhs)


@dataclass
class Vertex:
    """Vertices in the computational graph."""

    label: str
    data: ArrayF64