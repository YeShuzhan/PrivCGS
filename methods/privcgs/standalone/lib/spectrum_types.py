from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ReleasedSpectrum:
    probability: np.ndarray
    integer_histogram: np.ndarray
    edge_count: int
    endpoint_mass: int
    diagnostics: Mapping[str, object]
