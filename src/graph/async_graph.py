from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
import asyncio

import numpy as np
import numpy.typing as npt

from vsa.hrr import HRR

from threading import Thread


@dataclass
class Graph:
    connections: t.List[t.List[Connection]]

    async def run(self) -> None:
        # print("Running the graph")

        for index, depth in enumerate(self.connections):
            # print(f"running depth {index}")
            tasks = []
            for connection in depth:
                tasks.append(connection.feed())
            await asyncio.gather(*tasks)
            # print(f"completed depth {index}")


type Label = str
type ConnectionItem = t.Tuple[Label, Node]


@dataclass
class Connection:
    items: t.List[ConnectionItem]
    output_node: Node

    async def feed(self) -> None:
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
    label: Label
    input_buffers: t.Dict[Label, npt.NDArray[np.float64]]
    output_buffers: t.Dict[Label, npt.NDArray[np.float64]]

    @abstractmethod
    async def __call__(self, items: t.List[npt.NDArray[t.Any]]) -> None: ...

    @abstractmethod
    def __getitem__(self, key: Label) -> npt.NDArray[t.Any]: ...


class Bind(Node):
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
            self.input_buffers["lhs"][:] = inputs[0]
            self.input_buffers["rhs"][:] = inputs[1]
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
