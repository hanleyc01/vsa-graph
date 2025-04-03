import typing as t

from lark import Transformer, Tree, Token

import mincaml.syntax as syn
import mincaml.types as typ


type _AnyList = t.List[t.Any]


class Tree2Syntax(Transformer):

    def start(self, args: _AnyList) -> syn.Syntax:
        return args[0]

    def true_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Bool(bool(args[0]))

    def string_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.String(args[0].value)

    def ident(self, args: _AnyList) -> syn.Syntax:
        return syn.Ident(args[0].value)

    def get_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Get(args[0], args[1][0])

    def simple_expr(self, args: _AnyList) -> syn.Syntax:
        return args[0]

    def not_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Not(args[0])

    def if_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.If(args[0], args[1][0], args[2][0])

    def let_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Let((args[0], typ.gentyp()), args[1][0], args[2][0])

    def letrec_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.LetRec(
            syn.Fundef(
                (args[0], typ.gentyp()),
                [],
                args[1][0]
            ),
            args[2][0]
        )

    def tuple_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Tuple([arg[0] for arg in args])

    def seq_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.Seq([arg[0] for arg in args])

    def expr(self, args: _AnyList) -> syn.Syntax: 
        return args #type: ignore

    def app_expr(self, args: _AnyList) -> syn.Syntax:
        return syn.App(args[0], [arg.children for arg in args[1:]][0])


def transform(tree: Tree[t.Any]) -> syn.Syntax:
    """Transform the Tree into syntax data classes."""
    transformer = Tree2Syntax()
    return transformer.transform(tree)
