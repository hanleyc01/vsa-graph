"""Module the syntax of MinCaml."""

from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Type


class Syntax(ABC):
    """Abstract base class of MinCaml syntax objects."""

    pass


@dataclass
class Fundef:
    name: t.Tuple[Ident, Type]
    args: t.List[t.Tuple[Ident, Type]]
    body: Syntax


@dataclass
class Unit(Syntax):
    pass


@dataclass
class Bool(Syntax):
    value: bool


@dataclass
class String(Syntax):
    data: str


@dataclass
class Ident(Syntax):
    cont: str


@dataclass(init=False)
class Int(Syntax):
    value: int

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported!")


@dataclass(init=False)
class Float(Syntax):
    value: float

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass
class Not(Syntax):
    operand: Syntax


@dataclass(init=False)
class Neg(Syntax):
    operand: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class Add(Syntax):
    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class Sub(Syntax):
    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FNeg(Syntax):
    operand: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FAdd(Syntax):
    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FSub(Syntax):
    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FMul(Syntax):
    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass(init=False)
class FDiv(Syntax):
    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass
class Eq(Syntax):
    left: Syntax
    right: Syntax


@dataclass(init=False)
class LE(Syntax):
    left: Syntax
    right: Syntax

    def __init__(self) -> None:
        raise NotImplementedError("Numbers are not yet supported")


@dataclass
class If(Syntax):
    cond: Syntax
    then_expr: Syntax
    else_expr: Syntax


@dataclass
class Let(Syntax):
    binding: t.Tuple[Ident, Type]
    expr: Syntax
    body: Syntax


@dataclass
class LetRec(Syntax):
    fundef: Fundef
    body: Syntax


@dataclass
class App(Syntax):
    func: Syntax
    args: t.List[Syntax]


@dataclass
class Tuple(Syntax):
    elements: t.List[Syntax]


@dataclass
class Seq(Syntax):
    elements: t.List[Syntax]


@dataclass
class LetTuple(Syntax):
    bindings: t.List[t.Tuple[Ident, Type]]
    expr: Syntax
    body: Syntax


@dataclass
class Array(Syntax):
    size: Syntax
    init: Syntax


@dataclass
class Get(Syntax):
    array: Syntax
    index: Syntax


@dataclass
class Put(Syntax):
    array: Syntax
    index: Syntax
    value: Syntax
