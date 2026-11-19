from __future__ import annotations
import hashlib, heapq, json, math, os, time
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Mapping
import numpy as np


OUT = Path(__file__).resolve().parent / "artifacts"


PAIR_SEED = 1_517_938_221


DONOR_SLOTS = 4


PROGRESS_EVERY_PIVOTS = 10_000


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical(edges: np.ndarray) -> np.ndarray:
    value = np.asarray(edges, dtype=np.int64).reshape(-1, 2).copy()
    value.sort(axis=1)
    order = np.lexsort((value[:, 1], value[:, 0]))
    value = np.ascontiguousarray(value[order], dtype=np.int64)
    if len(value) and np.any(value[:, 0] == value[:, 1]):
        raise AssertionError("self loop")
    if len(value) > 1 and np.any(np.all(value[1:] == value[:-1], axis=1)):
        raise AssertionError("duplicate edge")
    return value


def save_edges(path: Path, edges: np.ndarray) -> np.ndarray:
    ordered = canonical(edges)
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savetxt(temporary, ordered, fmt="%d")
    os.replace(temporary, path)
    return ordered


def adjacency_from_edges(edges: np.ndarray, n_nodes: int) -> list[set[int]]:
    adjacency = [set() for _ in range(n_nodes)]
    for raw_left, raw_right in edges:
        left, right = int(raw_left), int(raw_right)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def common_count(adjacency: list[set[int]], left: int, right: int) -> int:
    first, second = adjacency[left], adjacency[right]
    if len(first) > len(second):
        first, second = second, first
    return sum(1 for node in first if node in second)


def count_triangles(edges: np.ndarray, adjacency: list[set[int]]) -> int:
    incidence = sum(
        common_count(adjacency, int(left), int(right)) for left, right in edges
    )
    if incidence % 3:
        raise AssertionError(f"triangle incidence remainder {incidence % 3}")
    return incidence // 3


def splitmix64_scalar(value: int, seed: int) -> int:
    mask = (1 << 64) - 1
    x = (int(value) + int(seed) + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return (x ^ (x >> 31)) & mask


def insert_donor(
    donor_index: np.ndarray,
    donor_support: np.ndarray,
    node: int,
    edge_index: int,
    support: int,
) -> None:
    indices = donor_index[node]
    scores = donor_support[node]
    if (support, edge_index) >= (int(scores[-1]), int(indices[-1])):
        return
    position = DONOR_SLOTS - 1
    while position > 0 and (support, edge_index) < (
        int(scores[position - 1]),
        int(indices[position - 1]),
    ):
        scores[position] = scores[position - 1]
        indices[position] = indices[position - 1]
        position -= 1
    scores[position] = support
    indices[position] = edge_index


def build_donor_table(
    edges: np.ndarray,
    adjacency: list[set[int]],
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_support = np.empty(len(edges), dtype=np.int64)
    donor_index = np.full((n_nodes, DONOR_SLOTS), -1, dtype=np.int64)
    donor_support = np.full(
        (n_nodes, DONOR_SLOTS), np.iinfo(np.int64).max, dtype=np.int64
    )
    for index, (raw_left, raw_right) in enumerate(edges):
        left, right = int(raw_left), int(raw_right)
        support = common_count(adjacency, left, right)
        edge_support[index] = support
        insert_donor(donor_index, donor_support, left, index, (0 if len(adjacency[right])==1 else 1 if len(adjacency[right])==2 else 2)*(n_nodes+1)+support)
        insert_donor(donor_index, donor_support, right, index, (0 if len(adjacency[left])==1 else 1 if len(adjacency[left])==2 else 2)*(n_nodes+1)+support)
    return edge_support, donor_index, donor_support


def other_endpoint(edges: np.ndarray, edge_index: int, node: int) -> int | None:
    left, right = map(int, edges[edge_index])
    if left == node:
        return right
    if right == node:
        return left
    return None


def evaluate_switch(
    adjacency: list[set[int]],
    removed_first: tuple[int, int],
    removed_second: tuple[int, int],
    added_first: tuple[int, int],
    added_second: tuple[int, int],
    commit: bool,
) -> tuple[int, bool]:
    a, u = removed_first
    b, v = removed_second
    x, y = added_first
    z, w = added_second
    if len({a, u, b, v}) != 4:
        return 0, False
    if x == y or z == w or {x, y} == {z, w}:
        return 0, False
    if u not in adjacency[a] or v not in adjacency[b]:
        return 0, False

    removed_support_first = common_count(adjacency, a, u)
    adjacency[a].remove(u)
    adjacency[u].remove(a)
    removed_support_second = common_count(adjacency, b, v)
    adjacency[b].remove(v)
    adjacency[v].remove(b)

    valid = (
        y not in adjacency[x]
        and w not in adjacency[z]
        and x != y
        and z != w
        and {x, y} != {z, w}
    )
    if not valid:
        adjacency[a].add(u)
        adjacency[u].add(a)
        adjacency[b].add(v)
        adjacency[v].add(b)
        return 0, False

    added_support_first = common_count(adjacency, x, y)
    adjacency[x].add(y)
    adjacency[y].add(x)
    added_support_second = common_count(adjacency, z, w)
    delta = (
        -removed_support_first
        - removed_support_second
        + added_support_first
        + added_support_second
    )
    if commit:
        adjacency[z].add(w)
        adjacency[w].add(z)
    else:
        adjacency[x].remove(y)
        adjacency[y].remove(x)
        adjacency[a].add(u)
        adjacency[u].add(a)
        adjacency[b].add(v)
        adjacency[v].add(b)
    return int(delta), True


def increase_sweep(dataset, edges, adjacency, target_degree, current, low, high, sweep, risk_policy=None):
    began = time.perf_counter()
    _, donor_index, _ = build_donor_table(edges, adjacency, len(target_degree))
    seed_value = (PAIR_SEED + sweep * 0x9E3779B1) & ((1 << 64) - 1)
    pivots = sorted(map(int, np.flatnonzero(target_degree >= 2)), key=lambda u: splitmix64_scalar(u, seed_value ^ 0xA24BAED4963EE407))
    attempted = accepted = gain = open_wedges = scanned = nonpositive = overshoot = 0
    kinds = {'1_1':0, '1_2':0, '2_2':0, 'one_low':0, 'other':0}
    for position, pivot in enumerate(pivots):
        if (risk_policy.hit(current) if risk_policy is not None else low <= current <= high): break
        neighbors = sorted(adjacency[pivot], key=lambda u: splitmix64_scalar(u, seed_value ^ ((pivot+1)*0xD6E8FEB86659FD93)))
        for cursor in range(0,len(neighbors)-1,2):
            if (risk_policy.hit(current) if risk_policy is not None else low <= current <= high): break
            a,b = map(int,neighbors[cursor:cursor+2])
            if b in adjacency[a]: continue
            open_wedges += 1
            candidates=[]
            for first in donor_index[a]:
                first=int(first)
                if first<0:continue
                u=other_endpoint(edges,first,a)
                if u is None or u==pivot or u==b:continue
                for second in donor_index[b]:
                    second=int(second);scanned+=1
                    if second<0 or first==second:continue
                    v=other_endpoint(edges,second,b)
                    if v is None or v==pivot or len({a,u,b,v})!=4 or v in adjacency[u]:continue
                    du,dv=int(target_degree[u]),int(target_degree[v])
                    support=common_count(adjacency,a,u)+common_count(adjacency,b,v)
                    rank=(-(int(du<=2)+int(dv<=2)),-(int(du==1)+int(dv==1)),support,first,second)
                    candidates.append((rank,first,second,u,v,du,dv))
            for _,first,second,u,v,du,dv in sorted(candidates):
                attempted+=1
                if risk_policy is not None: risk_policy.candidate_scans+=1
                delta,valid=evaluate_switch(adjacency,(a,u),(b,v),(a,b),(u,v),False)
                if not valid:continue
                if risk_policy is not None:
                    if not risk_policy.allow(current,current+delta):continue
                else:
                    if delta<=0:nonpositive+=1;continue
                    if current+delta>high:overshoot+=1;continue
                actual,valid=evaluate_switch(adjacency,(a,u),(b,v),(a,b),(u,v),True)
                assert valid and actual==delta
                edges[first]=(min(a,b),max(a,b));edges[second]=(min(u,v),max(u,v))
                current+=delta;gain+=delta;accepted+=1
                if risk_policy is not None: risk_policy.committed(current-delta,current)
                pair=tuple(sorted((du,dv)))
                key=f'{pair[0]}_{pair[1]}' if pair in ((1,1),(1,2),(2,2)) else ('one_low' if min(pair)<=2 else 'other')
                kinds[key]+=1
                if pair==(1,1):assert adjacency[u]=={v} and adjacency[v]=={u}
                break
        if position and position%PROGRESS_EVERY_PIVOTS==0:
            write_json(OUT/'progress.json',{'dataset':dataset,'phase':'low_degree_sweep','sweep':sweep,'pivots_completed':position,'accepted':accepted,'triangles':current,'accepted_degree_pairs':kinds})
    assert np.array_equal(np.bincount(edges.ravel(),minlength=len(target_degree)),target_degree)
    return current,{'sweep':sweep,'open_wedges':open_wedges,'candidate_pairs_scanned':scanned,'attempted_donor_pairs':attempted,'accepted':accepted,'triangle_gain':gain,'accepted_degree_pairs':kinds,'guaranteed_new_size2_components':kinds['1_1'],'rejected_nonpositive':nonpositive,'rejected_overshoot':overshoot,'interval_hit':bool(low<=current<=high),'sweep_seconds':time.perf_counter()-began}


def decrease_sweep(dataset, edges, adjacency, target_degree, current, low, high, sweep, risk_policy=None):
    began=time.perf_counter()
    support=np.asarray([common_count(adjacency,int(a),int(b)) for a,b in edges],dtype=np.int64)
    order=np.lexsort((np.arange(len(edges)),support))
    accepted=attempted=overshoot=nonnegative=0; gain=0; ledger=[]
    # Deterministic high-support donor paired with rotating low-support donors.
    # Existing DONOR_SLOTS is retained; no data-dependent tuning.
    for pos,first in enumerate(order[::-1]):
        if (risk_policy.hit(current) if risk_policy is not None else low<=current<=high):break
        first=int(first)
        a,b=map(int,edges[first])
        if common_count(adjacency,a,b)==0:continue
        success=False
        for slot in range(min(DONOR_SLOTS,len(edges))):
            second=int(order[(pos*DONOR_SLOTS+slot+sweep)%len(edges)])
            c,d=map(int,edges[second])
            if len({a,b,c,d})!=4:continue
            for added in (((a,c),(b,d)),((a,d),(b,c))):
                attempted+=1
                if risk_policy is not None:risk_policy.candidate_scans+=1
                delta,valid=evaluate_switch(adjacency,(a,b),(c,d),*added,False)
                if not valid:continue
                if risk_policy is not None:
                    if not risk_policy.allow(current,current+delta):continue
                else:
                    if delta>=0:nonnegative+=1;continue
                    if current+delta<low:overshoot+=1;continue
                actual,valid=evaluate_switch(adjacency,(a,b),(c,d),*added,True)
                assert valid and actual==delta
                edges[first]=sorted(added[0]);edges[second]=sorted(added[1])
                current+=delta;gain+=delta;accepted+=1;success=True
                if risk_policy is not None:risk_policy.committed(current-delta,current)
                ledger.append((a,b,c,d,*added[0],*added[1],delta))
                break
            if success:break
    assert np.array_equal(np.bincount(edges.ravel(),minlength=len(target_degree)),target_degree)
    OUT.mkdir(parents=True,exist_ok=True)
    np.save(OUT/f'decrease_ledger_{sweep:03d}.npy',np.asarray(ledger,dtype=np.int64).reshape(-1,9))
    return current,{'direction':'decrease','sweep':sweep,'accepted':accepted,'triangle_gain':gain,'attempted_donor_pairs':attempted,'rejected_nonnegative':nonnegative,'rejected_overshoot':overshoot,'interval_hit':bool(low<=current<=high),'sweep_seconds':time.perf_counter()-began}


def degree_only_sweep(dataset,edges,adjacency,target_degree,current,low,high,sweep):
    if current>high:
        return decrease_sweep(dataset,edges,adjacency,target_degree,current,low,high,sweep)
    return increase_sweep(dataset,edges,adjacency,target_degree,current,low,high,sweep)
