"""Faux main used as a scratch pad."""

import time
import asyncio
import numpy as np
import string

from vsa_graph.vsa.hrr import HRR
from vsa_graph.graph.async_graph import Bind, Graph, VecF64, Connection, Node, Add
from vsa_graph.graph.graph_tools import display_async_graph
from vsa_graph.mincaml.grammar import parse
from vsa_graph.mincaml.tree_transformer import transform
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
        VecF64(name, symbol.data) for name, symbol in codebook.items()
    ]

    levels = []

    lvl1 = []
    input_iter = iter(user_inputs)
    for lhs, rhs in zip(input_iter, input_iter):
        lvl1.append(
            Connection(
                [lhs, rhs], Bind(new_label("bind"), dim)
            )
        )
    levels.append(lvl1)

    while len(levels[-1]) > 1:
        lvl = []
        previous_level_iter = iter(item.output_node for item in levels[-1])
        for lhs, rhs in zip(previous_level_iter, previous_level_iter):
            lvl.append(
                Connection(
                    [lhs, rhs], Bind(new_label("bind"), dim)
                )
            )
        levels.append(lvl)

    return Graph(levels), codebook


async def async_main() -> None:
    dim = 100
    powers = 8
    cgraph, codebook = gen_powers_of_2(powers, dim)
    start = time.time()
    await cgraph.run()
    end = time.time()
    print(
        f"time for execution of iterated binding of {2**powers} items: {end-start}"
    )
    display_async_graph(cgraph)

test_expr = """
cons foo bar
"""


def main() -> None:
    test_str = "foo, bar"
    ast_tree = parse(test_expr)
    print(ast_tree.pretty())

    pprint(transform(ast_tree))


if __name__ == "__main__":
    asyncio.run(async_main())
    # main()
