import time
import asyncio
import numpy as np
import string

from vsa.hrr import HRR
from graph.async_graph import Bind, Graph, UserInput, Connection, Node, Add
import typing as t

from pprint import pprint


label_ctr = -1


def new_label(s: str) -> str:
    global label_ctr
    label_ctr += 1
    return f"{s}{label_ctr}"


def gen_powers_of_2(powers: int, dim: int) -> t.Tuple[Graph, t.Dict[str, HRR]]:
    vocab = [new_label("x") for _ in range(2**powers)]
    codebook = {name: HRR.normal(dim) for name in vocab}
    user_inputs: t.List[Node] = [
        UserInput(name, symbol.data) for name, symbol in codebook.items()
    ]

    levels = []

    lvl1 = []
    input_iter = iter(user_inputs)
    for lhs, rhs in zip(input_iter, input_iter):
        lvl1.append(
            Connection(
                [("out", lhs), ("out", rhs)], Add(new_label("add"), dim)
            )
        )
    levels.append(lvl1)

    while len(levels[-1]) > 1:
        lvl = []
        previous_level_iter = iter(item.output_node for item in levels[-1])
        for lhs, rhs in zip(previous_level_iter, previous_level_iter):
            lvl.append(
                Connection(
                    [("out", lhs), ("out", rhs)], Add(new_label("add"), dim)
                )
            )
        levels.append(lvl)

    return Graph(levels), codebook


async def main() -> None:
    dim = 100
    powers = 13
    cgraph, codebook = gen_powers_of_2(powers, dim)
    start = time.time()
    await cgraph.run()
    end = time.time()
    print(f"time for execution: {end-start}")

    last_out = cgraph.connections[-1][0].output_node["out"]
    print(last_out)
    print(HRR.similarity(codebook["x0"].data, last_out))


if __name__ == "__main__":
    asyncio.run(main())
