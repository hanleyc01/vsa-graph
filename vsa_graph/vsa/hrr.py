"""Implementation of Tony Plate's Holographic Reduced Representations (HRR).

For the reference implementation, see
(this link)[https://github.com/ecphory/hrr/blob/main/hrr/hrr.py].
"""

from __future__ import annotations

import math
import typing as t

import numpy as np
import numpy.typing as npt
from numpy.fft import fft, ifft

from . import vsa


class HRR(vsa.VSA[np.float64]):
    """Holographic reduced representation vectors.

    The vectors of HRR are sampled from a normal distribution. Implements
    binding through circular convolution.
    """

    data: npt.NDArray[np.float64]

    def __init__(self, data: npt.NDArray[np.float64]) -> None:
        self.data = data

    @staticmethod
    def normal(size: int, sd: float | None = None) -> HRR:
        if sd is None:
            sd = 1.0 / math.sqrt(size)
        data = np.random.normal(scale=sd, size=size)
        data /= np.linalg.norm(data)
        return HRR(data)

    @staticmethod
    def bind(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> np.typing.NDArray[np.float64]:
        return t.cast(npt.NDArray[np.float64], ifft(fft(x) * fft(y)).real)

    @staticmethod
    def bundle(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return x + y

    @staticmethod
    def inv(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return x[np.r_[0, x.size - 1 : 0 : -1]]

    @staticmethod
    def unbind(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        return HRR.bind(x, HRR.inv(y))

    @staticmethod
    def similarity(
        x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> float:
        return float(x.dot(y)) / x.size

    def __add__(self, rhs: HRR | float | int) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.bundle(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data + rhs)
        elif isinstance(rhs, float):
            return HRR(self.data + rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __radd__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(self.data + rhs.data)
        elif isinstance(rhs, int):
            return HRR(self.data + rhs)
        elif isinstance(rhs, float):
            return HRR(self.data + rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __sub__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(self.data - rhs.data)
        elif isinstance(rhs, int):
            return HRR(self.data - rhs)
        elif isinstance(rhs, float):
            return HRR(self.data - rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __mul__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.bind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data * rhs)
        elif isinstance(rhs, float):
            return HRR(self.data * rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __rmul__(self, rhs: int | float | HRR) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.bind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data * rhs)
        elif isinstance(rhs, float):
            return HRR(self.data * rhs)
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __truediv__(self, rhs: HRR | int | float) -> HRR:
        if isinstance(rhs, HRR):
            return HRR(HRR.unbind(self.data, rhs.data))
        elif isinstance(rhs, int):
            return HRR(self.data / rhs)
        elif isinstance(rhs, float):
            return HRR((self.data / rhs).astype(np.float64))
        else:
            raise TypeError(f"Innapropriate argument type: {type(rhs)}")

    def __invert__(self) -> HRR:
        return HRR(HRR.inv(self.data))

    def __neg__(self) -> HRR:
        return HRR(-self.data)

    def magnitude(self) -> float:
        return math.sqrt(self.data @ self.data) / self.data.size

    def __matmul__(
        self, other: HRR | npt.NDArray[np.float64]
    ) -> float | npt.NDArray[np.float64]:
        if isinstance(other, HRR):
            return self.data @ other.data
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            if len(other.shape) == 2:
                return (self.data @ other).astype(np.float64)
            else:
                return self.data @ other
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def sim(self, other: HRR | npt.NDArray[np.float64]) -> float:
        if isinstance(other, HRR):
            return HRR.similarity(self.data, other.data)
        elif isinstance(other, np.ndarray) and other.dtype == np.float64:
            return HRR.similarity(self.data, other)
        else:
            raise TypeError(f"Innapropriate argument type {type(other)}")

    def __str__(self) -> str:
        return f"HRR({self.data})"
