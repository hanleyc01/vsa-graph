"""Module the syntax of MinCaml.

For more information about the exact syntactic specification, see:
[the MinCaml website](https://esumii.github.io/min-caml/index-e.html).
"""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Type


class Syntax(ABC):
    """Abstract base class of MinCaml syntax objects.

    MinCaml is a subset of ML. Except for some details such as priority and parentheses, it has the following syntax.

    e ::=	                            expressions
        c	                            constants
        op(e1, ..., en)	                primitive operations
        if e1 then e2 else e3	        conditional branches
        let x = e1 in e2	            variable definitions
        x	                            variables
        let rec x y1 ... yn = e1 in e2	function definitions (mutually recursive)
        e e1 ... en	                    function applications
        (e1, ..., en)	                tuple creations
        let (x1, ..., xn) = e1 in e2	read from tuples
        Array.create e1 e2	            array creations
        e1.(e2)	                        read from arrays
        e1.(e2) <- e3	                write to arrays
    """

    pass


@dataclass
class Fundef:
    """Declaration body of a `let rec ...` expression."""

    name: t.Tuple[Ident, Type]
    args: t.List[t.Tuple[Ident, Type]]
    body: Syntax


@dataclass
class Unit(Syntax):
    """The unit constructor."""

    pass


@dataclass
class Bool(Syntax):
    """Booleans values."""

    value: bool


@dataclass
class String(Syntax):
    """String values."""

    data: str


@dataclass
class Ident(Syntax):
    """Identifiers."""

    cont: str


@dataclass(init=False)
class Int(Syntax):
    """Integers: not currently supported."""

    value: int

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported!")


@dataclass(init=False)
class Float(Syntax):
    """Floats: not currently supported"""

    value: float

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass
class Not(Syntax):
    """Logical negation."""

    operand: Syntax


@dataclass(init=False)
class Neg(Syntax):
    """Integer negation: not currently supported."""

    operand: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class Add(Syntax):
    """Integer addition: not currently supported."""

    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class Sub(Syntax):
    """Integer subtraction: not currently supported."""

    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FNeg(Syntax):
    """Float negation: not currently supported."""

    operand: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FAdd(Syntax):
    """Float addition: not currently supported."""

    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FSub(Syntax):
    """Float subtraction: not currently supported."""

    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FMul(Syntax):
    """Float multiplication: not currently supported."""

    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FDiv(Syntax):
    """Float division: not currently supported."""

    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass
class Eq(Syntax):
    """Logical equation, currently only for boolean values."""

    left: Syntax
    right: Syntax


@dataclass(init=False)
class LE(Syntax):
    """Integer comparison: not currently supported."""

    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass
class If(Syntax):
    """If expressions."""

    cond: Syntax
    then_expr: Syntax
    else_expr: Syntax


@dataclass
class Let(Syntax):
    """Let expressions."""

    binding: t.Tuple[Ident, Type]
    expr: Syntax
    body: Syntax


@dataclass
class LetRec(Syntax):
    """Let recursive expressions, function definitions."""

    fundef: Fundef
    body: Syntax


@dataclass
class App(Syntax):
    """Function application."""

    rator: Syntax
    rand: t.List[Syntax]


@dataclass
class Tuple(Syntax):
    """Tuple constructor."""

    elements: t.List[Syntax]


@dataclass
class Seq(Syntax):
    """List constructor."""

    elements: t.List[Syntax]


@dataclass
class LetTuple(Syntax):
    """Pattern deconstruction."""

    bindings: t.List[t.Tuple[Ident, Type]]
    expr: Syntax
    body: Syntax


@dataclass(init=False)
class Array(Syntax):
    """Array initialization: not currently supported."""

    size: Syntax
    init: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Not currently supported.")


@dataclass
class Get(Syntax):
    """Array projection."""

    array: Syntax
    index: Syntax


@dataclass
class Put(Syntax):
    """Array injection."""

    array: Syntax
    index: Syntax
    value: Syntax
