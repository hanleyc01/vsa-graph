"""The source-file level syntax of the MinCaml programming language, sine
mathematical operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import typing as t
import sys

import logging
from lark import Lark, Tree, logger


logger.setLevel(logging.WARN)


# Grammar of the subset of the language. Should mirror `./grammar.lark`.
_GRAMMAR = """
// simple grammar for reference specification, see 
// https://esumii.github.io/min-caml/index-e.html

start: expr

expr: simple_expr
    | "not" expr -> not_expr
    | "if" expr "then" expr "else" expr -> if_expr
    | "let" ident "=" expr "in" expr -> let_expr
    | "let" "rec" ident "=" expr "in" expr -> letrec_expr
    | expr app_arguments+ -> app_expr
    | "let" "(" pat ")" "=" expr "in" expr -> let_pattern_expr
    | expr ("," expr)+ -> tuple_expr
    | expr (";" expr)+ -> seq_expr

?simple_expr: "true" -> true_expr
    | "false" -> false_expr
    | ESCAPED_STRING -> string_expr
    | ident
    | "(" ")" -> unit_expr
    | simple_expr "." "(" expr ")" -> get_expr
    | "(" expr ")"


app_arguments: simple_expr+

pat: ident ("," ident)+

ident: CNAME "_"
    | CNAME

%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.CNAME
%import common.WS
%import common.DIGIT


%ignore WS
"""


# Lark parser generated from the provided grammar.
_parser = Lark(grammar=_GRAMMAR)


def parse(text: str) -> Tree[t.Any]:
    """Parse a string into a `lark.Tree` representation."""
    tree = _parser.parse(text)
    return tree
