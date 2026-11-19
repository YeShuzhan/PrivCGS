from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


def spawn_release_streams(seed):
    # Children 0/1 exactly match the frozen degree/operator streams. Child 2 is
    # reserved for the historical closure release and is not consumed here.
    children=np.random.SeedSequence(int(seed)).spawn(5)
    return tuple(np.random.default_rng(children[k]) for k in (0,1,3,4))
