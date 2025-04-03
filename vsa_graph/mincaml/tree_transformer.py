import typing as t

from lark import Transformer, Tree

import mincaml.syntax as syn


def transform(tree: Tree[t.Any]) -> syn.Syntax:
    """Transform the Tree into syntax data classes."""
    raise Exception("todo")
