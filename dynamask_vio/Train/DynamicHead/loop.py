from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class SequenceWindowSampler:
    length: int
    window_len: int
    stride: int | None = None

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.window_len <= 0:
            raise ValueError("window_len must be positive")
        if self.window_len > self.length:
            raise ValueError("window_len must not exceed sequence length")
        stride = self.window_len if self.stride is None else self.stride
        if stride <= 0:
            raise ValueError("stride must be positive")
        object.__setattr__(self, "stride", stride)

    def __iter__(self) -> Iterator[list[int]]:
        assert self.stride is not None
        start = 0
        while start + self.window_len <= self.length:
            yield list(range(start, start + self.window_len))
            start += self.stride

    def __len__(self) -> int:
        assert self.stride is not None
        return max(0, (self.length - self.window_len) // self.stride + 1)
