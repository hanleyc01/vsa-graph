import typing as t

from lark import Transformer, Tree, Token


import vsa_graph.mincaml.syntax as syn
import vsa_graph.mincaml.types as typ


type _AnyList = t.List[t.Any]


def _get_expr(x: t.Any) -> syn.Syntax:
    if isinstance(x, list):
        return _get_expr(x[0])
    elif isinstance(x, syn.Syntax):
        return x
    else:
        raise TypeError(f"Unknown type {x}")


class Tree2Syntax(Transformer[t.Any, syn.Syntax]):

    def start(self, args: _AnyList) -> syn.Syntax:
        return t.cast(syn.Syntax, args[0])

    def true_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Bool(bool(args[0]))

    def string_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.String(args[0].value)

    def ident(self, args: _AnyList) -> syn.Syntax:
        return syn.Ident(args[0].value)

    def get_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Get(_get_expr(args[0]), _get_expr(args[1]))

    def simple_expr(self, args: _AnyList) -> syn.Syntax:
        return _get_expr(args[0])

    def not_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Not(_get_expr(args[0]))

    def if_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.If(args[0], _get_expr(args[1]), _get_expr(args[2]))

    def let_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Let(
            (args[0], typ.gentyp()), _get_expr(args[1]), _get_expr(args[2])
        )

    def letrec_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.LetRec(
            syn.Fundef((args[0], typ.gentyp()), [], _get_expr(args[1])),
            _get_expr(args[2]),
        )

    def tuple_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Tuple([_get_expr(arg[0]) for arg in args])

    def seq_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Seq([_get_expr(arg[0]) for arg in args])

    def expr(self, args: _AnyList) -> syn.Syntax:
        return args  # type: ignore

    def app_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.App(args[0], [arg.children for arg in args[1:]][0])


def transform(tree: Tree[t.Any]) -> syn.Syntax:
    """Transform the Tree into syntax data classes."""
    transformer = Tree2Syntax()
    return transformer.transform(tree)
