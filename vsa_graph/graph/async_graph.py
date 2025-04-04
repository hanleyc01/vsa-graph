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
type ConnectionItem = t.Tuple[Label, Node]
"""A `ConnectionItem` contains label name for the output buffer of the 
incoming node to a connection, and the Node itself.
"""


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

    items: t.List[ConnectionItem]
    output_node: Node

    async def feed(self) -> None:
        """Feed the buffers from nodes given in `self.items` to
        `self.output_node.__call__`.
        """
        # print("feeding inputs to output node")
        input_buffers = []
        for buffer_label, input_node in self.items:
            # print(f"getting {buffer_label} from {input_node.label}")
            # print(f"{input_node[buffer_label] = }")
            input_buffers.append(input_node[buffer_label])

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
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    """
    Named input buffers, which hold immutable references to buffers from
    other nodes.
    """
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    """The output buffer which is the result of `__call__`.
    """

    @abstractmethod
    async def __call__(self, items: t.List[npt.NDArray[t.Any]]) -> None:
        """Perform some operation over `items` in order."""
        ...

    @abstractmethod
    def __getitem__(self, key: Label) -> npt.NDArray[t.Any]:
        """Search the Node's buffers for some buffer associated with `key`."""


class Bind(Node):
    """A HRR binding node.

    HRR binding is implemented through circular convolution of the input
    buffers, outputting the result in `self.output_buffer["out"]`.
    """

    label: Label
    dim: int
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]

    def __init__(self, label: str, dim: int) -> None:
        self.label = label
        self.dim = dim

        self.input_buffers = {"lhs": np.zeros(dim), "rhs": np.zeros(dim)}
        self.output_buffers = {"out": np.zeros(dim)}

    async def __call__(self, inputs: t.List[npt.NDArray[np.float64]]) -> None:
        assert len(inputs) == 2, "nargs of bind == 2"
        for item in inputs:
            assert (
                self.dim == item.size
            ), f"expected dim: {self.dim} got {item.size}"

        def bind() -> npt.NDArray[np.float64]:
            self.input_buffers["lhs"] = inputs[0]
            self.input_buffers["rhs"] = inputs[1]
            return HRR.bind(
                self.input_buffers["lhs"], self.input_buffers["rhs"]
            )

        self.output_buffers["out"] = (
            await asyncio.get_event_loop().run_in_executor(None, bind)
        )

        # print(f"for {self.label}, {self.output_buffers["out"] =}")

        self.__post_call__()

    def __post_call__(self) -> None:
        for key in self.input_buffers.keys():
            self.input_buffers[key] = np.zeros(self.dim)

    def __repr__(self) -> str:
        return f"Bind(label: {self.label})"

    def __getitem__(self, key: Label) -> npt.NDArray[np.float64]:
        input_buff = self.input_buffers.get(key)
        if input_buff is None:
            return self.output_buffers[key]
        else:
            return input_buff


class Add(Node):
    """A HRR bundle node.

    HRR bundling is implemented by element-wise addition between the
    input nodes, and put in `self.output_buffers["out"]`.
    """

    label: Label
    dim: int
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]

    def __init__(self, label: Label, dim: int) -> None:
        self.label = label
        self.dim = dim
        self.input_buffers = {
            "lhs": np.zeros(dim),
            "rhs": np.zeros(dim),
        }
        self.output_buffers = {
            "out": np.zeros(dim),
        }

    async def __call__(self, inputs: t.List[npt.NDArray[np.float64]]) -> None:
        assert len(inputs) == 2, "nargs of bind == 2"
        for item in inputs:
            assert (
                self.dim == item.size
            ), f"expected dim: {self.dim} got {item.size}"

        def add() -> npt.NDArray[np.float64]:
            self.input_buffers["lhs"][:] = inputs[0]
            self.input_buffers["rhs"][:] = inputs[1]
            return HRR.bundle(
                self.input_buffers["lhs"], self.input_buffers["rhs"]
            )

        self.output_buffers["out"] = (
            await asyncio.get_event_loop().run_in_executor(None, add)
        )

        self.__post_call__()

    def __post_call__(self) -> None:
        for key in self.input_buffers.keys():
            self.input_buffers[key] = np.zeros(self.dim)

    def __getitem__(self, key: Label) -> npt.NDArray[np.float64]:
        input_buff = self.input_buffers.get(key)
        if input_buff is None:
            return self.output_buffers[key]
        else:
            return input_buff


class UserInput(Node):
    """User input computational graph node.

    Used for prespecifying inputs to a computational graph.
    """

    label: Label
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    dim: int

    def __init__(self, label: str, input: npt.NDArray[t.Any]) -> None:
        self.label = label
        self.input_buffers = {}
        self.output_buffers = {"out": input}
        self.dim = input.size

    async def __call__(self, items: t.List[npt.NDArray[t.Any]]) -> None:
        raise NotImplementedError("bad")

    def __getitem__(self, key: Label) -> npt.NDArray[t.Any]:
        input_buff = self.input_buffers.get(key)
        if input_buff is None:
            return self.output_buffers[key]
        else:
            return input_buff
