"""Types of the MinCaml language.

For more information about the specification, see
the [MinCaml website](https://esumii.github.io/min-caml/index-e.html).

T ::=	                    types
    π	                    primitive types
    T1 -> ... -> Tn -> T	function types
    T1 * ... * Tn	        tuple types
    T array	                array types
    α	                    type variables
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


class Type(ABC):
    """Abstract base class of types."""

    pass


@dataclass
class Unit(Type):
    """The Unit type."""

    pass


@dataclass
class Bool(Type):
    """The Boolean type."""

    pass


@dataclass
class Int(Type):
    """Type of integers."""

    pass


@dataclass
class Float(Type):
    """Type of floats."""

    pass


@dataclass
class Fun(Type):
    """Type of functions."""

    args: List[Type]
    ret: Type


@dataclass
class Tuple(Type):
    """Type of tuples."""

    elements: List[Type]


@dataclass
class Array(Type):
    """Type of arrays."""

    element_type: Type


@dataclass
class Var(Type):
    """Type variables, used for inference."""

    contents: Optional[Type]


def gentyp() -> Var:
    """Initialize an empty type variable. Set to `None` before inference,
    where new type variables are synthesized.
    """
    return Var(None)
