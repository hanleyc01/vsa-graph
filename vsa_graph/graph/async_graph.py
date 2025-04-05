from __future__ import annotations

import asyncio
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread

import numpy as np
import numpy.typing as npt
from vsa_graph.vsa.hrr import HRR


@dataclass
class Graph:
    """Asynchronous Graph of edges between nodes.

    Args:
        connections (list[list[Connection]]): Jagged list for graph depth.
            Connections at the same depth will be called concurrently.
    """

    connections: t.List[t.List[Connection]]

    async def run(self) -> None:
        """Feeds the specified buffer labels of the input nodes to the output
        node's `__call__` method as a list.
        """
        # print("Running the graph")

        for index, depth in enumerate(self.connections):
            # print(f"running depth {index}")
            tasks = []
            for connection in depth:
                tasks.append(connection.feed())
            await asyncio.gather(*tasks)
            # print(f"completed depth {index}")


type Label = str
"""Type alias for `str`"""


@dataclass
class Connection:
    """Edge between many input `Node`'s and an output `Node`.

    Args:
        items (list[tuple[Label, Node]]): A list of tuples with the first
            member being the name of the buffer to pass to output and the
            second being a reference to the Node itself. Note that items
            are passed in the same order to `self.output_node.__call__`, in
            order to support non-symmetric operations.
        output_node (Node): The output node. Input node buffers are fed
            as arguments to `self.output_node.__call__` method.
    """

    input_nodes: t.List[Node]
    output_node: Node

    async def feed(self) -> None:
        """Feed the buffers from nodes given in `self.items` to
        `self.output_node.__call__`.
        """
        # print("feeding inputs to output node")
        input_buffers = []
        for node in self.input_nodes:
            # print(f"getting {buffer_label} from {input_node.label}")
            # print(f"{input_node[buffer_label] = }")
            input_buffers.append(node.output_buffer)

        # print(
        #     f"calling {self.output_node.label} with nargs {len(input_buffers)}"
        # )
        await self.output_node(input_buffers)


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
    output_buffer: npt.NDArray[t.Any]
    """The output buffer which is the result of `__call__`.
    """

    @abstractmethod
    async def __call__(self, items: t.List[npt.NDArray[t.Any]]) -> None:
        """Perform some operation over `items` in order."""
        ...


class Bind(Node):
    """A HRR binding node.

    HRR binding is implemented through circular convolution of the input
    buffers, outputting the result in `self.output_buffer["out"]`.
    """

    label: Label
    dim: int
    output_buffer: npt.NDArray[np.float64]

    def __init__(self, label: str, dim: int) -> None:
        self.label = label
        self.dim = dim

        self.output_buffer = np.zeros(dim)

    async def __call__(self, inputs: t.List[npt.NDArray[np.float64]]) -> None:
        assert len(inputs) == 2, "nargs of bind == 2"
        for item in inputs:
            assert (
                self.dim == item.size
            ), f"expected dim: {self.dim} got {item.size}"

        def bind() -> npt.NDArray[np.float64]:
            return HRR.bind(inputs[0], inputs[1])

        self.output_buffer = await asyncio.get_event_loop().run_in_executor(
            None, bind
        )

        # print(f"for {self.label}, {self.output_buffers["out"] =}")

    def __repr__(self) -> str:
        return f"Bind(label: {self.label})"


class Add(Node):
    """A HRR bundle node.

    HRR bundling is implemented by element-wise addition between the
    input nodes, and put in `self.output_buffers["out"]`.
    """

    label: Label
    dim: int
    output_buffer: npt.NDArray[np.float64]

    def __init__(self, label: Label, dim: int) -> None:
        self.label = label
        self.dim = dim
        self.output_buffer = np.zeros(dim)

    async def __call__(self, inputs: t.List[npt.NDArray[np.float64]]) -> None:
        assert len(inputs) == 2, "nargs of bind == 2"
        for item in inputs:
            assert (
                self.dim == item.size
            ), f"expected dim: {self.dim} got {item.size}"

        def add() -> npt.NDArray[np.float64]:
            return HRR.bundle(inputs[0], inputs[1])

        self.output_buffer = (
            await asyncio.get_event_loop().run_in_executor(None, add)
        )


class Vec[T: t.Any](Node):
    """Vector of type `T`.

    Used for prespecifying inputs to a computational graph.
    """

    label: Label
    output_buffer: npt.NDArray[T]
    dim: int

    def __init__(self, label: str, input: npt.NDArray[T]) -> None:
        self.label = label
        self.output_buffer = input
        self.dim = input.size

    async def __call__(self, items: t.List[npt.NDArray[T]]) -> None:
        raise NotImplementedError("bad")



VecF64 = Vec[np.float64]
"""Vector node of type `np.float64`."""