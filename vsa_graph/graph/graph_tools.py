"""Tools for displaying computational graphs.

Given that `Graph` types have no guarantees about looping, we cannot
do traditional topological sort.

TODO: investigate cyclic graph display procedures.
"""

import typing as t

from .sync_graph import Connection, Graph, Node


def display_graph(graph: Graph) -> None:
    """Display a computational graph in a user-friendly way."""
    raise NotImplementedError("Not yet supported")
