from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
from random import SystemRandom
import numpy as np


def spawn_release_streams(seed):
    # Children 0/1 exactly match the frozen degree/operator streams. Child 2 is
    # reserved for the historical closure release and is not consumed here.
    children=np.random.SeedSequence(int(seed)).spawn(5)
    return tuple(np.random.default_rng(children[k]) for k in (0,1,3,4))


class PrivateGaussian:
    """Gaussian draws from OS randomness, with no caller-supplied seed."""

    def __init__(self):
        self._rng = SystemRandom()

    def normal(self, loc, scale, size):
        shape = (size,) if isinstance(size, int) else tuple(size)
        count = math.prod(shape)
        return np.fromiter(
            (self._rng.normalvariate(loc, scale) for _ in range(count)),
            dtype=np.float64, count=count,
        ).reshape(shape)
