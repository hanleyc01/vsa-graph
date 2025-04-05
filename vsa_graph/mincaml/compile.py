"""Compilation of `.syntax.Syntax` to our computational graph."""

import vsa_graph.mincaml.syntax as syn
import vsa_graph.mincaml.types as typ
import vsa_graph.graph.async_graph as async_graph
import vsa_graph.vsa.hrr as hrr


def compile(syntax: syn.Syntax) -> async_graph.Graph:
    """Translate `MinCaml` syntax into a computational graph, obeying
    the semantics of the language.
    """
    raise Exception("")
