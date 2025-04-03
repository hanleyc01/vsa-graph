"""Module `vsa`.

Module defining the abstract base class of `VSA`s.
"""

import typing as t
from abc import ABCMeta, abstractmethod

import numpy.typing as npt


class VSA[T: t.Any](metaclass=ABCMeta):
    """Abstract base class of all VSA implementations.

    We define this class in order
    """

    @classmethod
    @abstractmethod
    def bind(cls, x: npt.NDArray[T], y: npt.NDArray[T]) -> npt.NDArray[T]: ...

    @classmethod
    @abstractmethod
    def bundle(
        cls, x: npt.NDArray[T], y: npt.NDArray[T]
    ) -> npt.NDArray[T]: ...

    @classmethod
    @abstractmethod
    def unbind(
        cls, x: npt.NDArray[T], y: npt.NDArray[T]
    ) -> npt.NDArray[T]: ...

    @classmethod
    @abstractmethod
    def similarity(cls, x: npt.NDArray[T], y: npt.NDArray[T]) -> float: ...
