"""Tools for displaying computational graphs.

Given that `Graph` types have no guarantees about looping, we cannot
do traditional topological sort.

TODO: investigate cyclic graph display procedures.
"""

import typing as t

import vsa_graph.graph.async_graph as async_graph
import vsa_graph.graph.async_graph as sync_graph


def display_async_graph(graph: async_graph.Graph) -> None:
    for depth, layer in enumerate(graph.connections):
        print(f"Depth: {depth}")
        for connection in layer:
            inputs = ", ".join([node.label for node in connection.input_nodes])
            print(f"    {inputs} -> {connection.output_node.label}")
