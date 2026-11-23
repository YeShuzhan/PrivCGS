"""Fixed public schedule, ideal-Gaussian edge-zCDP accounting.

No node-minibatch privacy amplification is claimed. A jointly clipped whole-
graph gradient query of L2 radius C has sensitivity at most 2*C.
Floating point/PRNG implementation is NOT an information-theoretic sampler proof.
"""
from dataclasses import dataclass, field
import math


def positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return value


def to_rho(epsilon, delta=1e-5):
    e = positive(epsilon, "epsilon")
    d = positive(delta, "delta")
    if d >= 1:
        raise ValueError("delta must be below 1")
    L = -math.log(d)
    return (e / (math.sqrt(L + e) + math.sqrt(L))) ** 2


def to_epsilon(rho, delta=1e-5):
    r = float(rho)
    if not math.isfinite(r) or r < 0:
        raise ValueError("rho must be nonnegative and finite")
    d = positive(delta, "delta")
    if d >= 1:
        raise ValueError("delta must be below 1")
    return r + 2 * math.sqrt(r * -math.log(d))


def gaussian(sensitivity, rho):
    delta2 = positive(sensitivity, "sensitivity")
    cap = positive(rho, "rho")
    sigma = delta2 / math.sqrt(2 * cap)
    # Round noise scale upward; compute and debit its actual advertised bound.
    sigma = math.nextafter(math.nextafter(sigma, math.inf), math.inf)
    cost = (delta2 / sigma) ** 2 / 2
    if not math.isfinite(sigma) or not 0 < cost <= cap:
        raise ValueError("unrepresentable Gaussian calibration")
    return sigma, cost


@dataclass
class Ledger:
    epsilon: float
    delta: float = 1e-5
    events: list = field(default_factory=list)

    def __post_init__(self):
        # The public guard margin is accounting conservatism, not relaxed DP.
        self.limit = to_rho(self.epsilon, self.delta) * (1 - 1e-10)

    @property
    def spent(self):
        return math.fsum(x["rho"] for x in self.events)

    def reserve(self, name, rho):
        cost = positive(rho, "rho charge")
        proposed = math.fsum([x["rho"] for x in self.events] + [cost])
        if proposed > self.limit or to_epsilon(proposed, self.delta) > self.epsilon:
            raise RuntimeError("privacy budget denied BEFORE query/update/release")
        self.events.append({"name": name, "rho": cost})

    def report(self):
        e = to_epsilon(self.spent, self.delta)
        assert self.spent <= self.limit and e <= self.epsilon
        return {"requested_epsilon": self.epsilon, "delta": self.delta,
            "rho_limit": self.limit, "rho_spent": self.spent,
            "epsilon_upper_bound": e, "within_budget": True,
            "events": self.events, "accounting": "adaptive composition of full-query Gaussian mechanisms; no subsampling amplification"}
