from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


def validate_graph(edges, target):
    assert len(edges)*2 == int(target.sum())
    assert np.array_equal(np.bincount(edges.ravel(), minlength=len(target)), target)
    assert np.all(edges[:,0] < edges[:,1])
    assert len(np.unique(edges, axis=0)) == len(edges)


def minimax_target(lower, upper, W, lo, hi):
    if not (0 < lower <= upper) or W == 0:
        return None
    center = 2*lower*upper/(lower+upper)
    candidates = {max(lo,min(hi,int(f(center*W/3)))) for f in (math.floor,math.ceil)}
    def risk(t):
        c = 3*t/W
        return max(abs(c-lower)/lower, abs(c-upper)/upper)
    target = min(candidates, key=lambda t:(risk(t), t))
    return target, center, risk(target)
