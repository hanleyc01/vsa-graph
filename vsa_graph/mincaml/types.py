from abc import ABC
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


class Type(ABC):
    pass


@dataclass
class Unit(Type):
    pass


@dataclass
class Bool(Type):
    pass


@dataclass
class Int(Type):
    pass


@dataclass
class Float(Type):
    pass


@dataclass
class Fun(Type):
    args: List[Type]
    ret: Type


@dataclass
class Tuple(Type):
    elements: List[Type]


@dataclass
class Array(Type):
    element_type: Type


@dataclass
class Var(Type):
    contents: Optional[Type]


def gentyp() -> Var:
    return Var(None)
