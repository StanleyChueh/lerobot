"""Synthetic-only telemetry contract. NOT the OpenArm TELEMETRY_SCHEMA.md.

Deliberately outside src/lerobot: no registration, training hook, or robot connection.
The final schema must define clock, acquisition/arrival semantics, units, joint IDs,
calibration and validity before a production adapter is implemented.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
PROVISIONAL = "synthetic-fixture-v0"


@dataclass(frozen=True)
class Contract:
    """Explicit synthetic joint identities and units."""

    names: tuple[str, ...]
    position_units: tuple[str, ...]
    velocity_units: tuple[str, ...]
    effort_units: tuple[str, ...]
    version: str = PROVISIONAL

    def __post_init__(self) -> None:
        """Reject incomplete or production contracts."""
        if self.version != PROVISIONAL:
            raise ValueError("Only the provisional synthetic contract is supported")
        if len(self.names) != 16 or len(set(self.names)) != 16:
            raise ValueError("Expected 16 unique joint names")
        if any(len(u) != 16 for u in (self.position_units, self.velocity_units, self.effort_units)):
            raise ValueError("Units must be declared per joint")


@dataclass(frozen=True)
class Packet:
    """One synthetic telemetry acquisition with availability time."""

    episode: str  # globally qualified dataset-revision/episode identity
    acquired_s: float
    available_s: float  # same provisional monotonic clock as observations
    contract: Contract
    velocity: tuple[float, ...]
    effort: tuple[float, ...]  # actuator-reported effort, never Cartesian force


@dataclass(frozen=True)
class Aligned:
    """Causal feature payload with freshness status."""

    values: Array  # velocity then effort, in target contract order
    valid: bool
    age_s: float


def align(
    packets: list[Packet],
    *,
    episode: str,
    observation_s: float,
    action_s: float,
    target: Contract,
    max_age_s: float,
    max_action_delay_s: float,
) -> Aligned:
    """Causal previous-sample join; stale/missing -> zero payload plus invalid mask.

    action_s is the separately logged supervised target time. It cannot make future
    telemetry available. No interpolation, cross-episode history, or inferred delay.
    """
    if not np.isfinite([observation_s, action_s, max_age_s, max_action_delay_s]).all():
        raise ValueError("Nonfinite timing")
    if max_age_s < 0 or not 0 <= action_s - observation_s <= max_action_delay_s:
        raise ValueError("Invalid action/observation timing")
    eligible = []
    for packet in packets:
        if packet.episode != episode:
            continue
        if not np.isfinite([packet.acquired_s, packet.available_s]).all():
            raise ValueError("Nonfinite packet timing")
        if packet.available_s < packet.acquired_s:
            raise ValueError("Availability precedes acquisition")
        if packet.acquired_s <= observation_s and packet.available_s <= observation_s:
            eligible.append(packet)
    if not eligible:
        return Aligned(np.zeros(32), False, float("inf"))
    packet = max(eligible, key=lambda p: (p.acquired_s, p.available_s))
    if sum(p.acquired_s == packet.acquired_s for p in eligible) > 1:
        raise ValueError("Ambiguous duplicate acquisition timestamp")
    if set(packet.contract.names) != set(target.names):
        raise ValueError("Joint identity mismatch")
    order = [packet.contract.names.index(n) for n in target.names]
    for source_units, target_units in zip(
        (packet.contract.position_units, packet.contract.velocity_units, packet.contract.effort_units),
        (target.position_units, target.velocity_units, target.effort_units),
        strict=True,
    ):
        if tuple(source_units[i] for i in order) != target_units:
            raise ValueError("Unit conversion requires a verified calibration")
    raw = np.asarray([packet.velocity, packet.effort], dtype=np.float64)
    if raw.shape != (2, 16) or not np.isfinite(raw).all():
        raise ValueError("Malformed or nonfinite telemetry")
    age = observation_s - packet.acquired_s
    return Aligned(
        raw[:, order].reshape(32).copy() if age <= max_age_s else np.zeros(32), age <= max_age_s, age
    )


@dataclass(frozen=True)
class TrainNormalizer:
    """Statistics fitted exclusively to declared training episodes."""

    mean: Array
    scale: Array
    training_ids: frozenset[str]

    @classmethod
    def fit(cls, rows: dict[str, Array], *, train_ids: set[str], test_ids: set[str]) -> "TrainNormalizer":
        """Fit valid rows only after checking episode isolation."""
        if train_ids & test_ids or set(rows) != train_ids or not train_ids:
            raise ValueError("Fit requires exactly the disjoint training episodes")
        arrays = [np.asarray(rows[k], dtype=np.float64) for k in sorted(train_ids)]
        if any(a.ndim != 2 or a.shape[1] != 32 or not len(a) for a in arrays):
            raise ValueError("Expected nonempty (frames,32) valid telemetry per episode")
        values = np.concatenate(arrays)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite training features")
        return cls(values.mean(0), np.maximum(values.std(0), 1e-8), frozenset(train_ids))

    def transform(self, row: Aligned) -> Array:
        """Normalize a valid aligned sample without updating statistics."""
        if not row.valid:
            raise ValueError("Missing telemetry must not enter a normalizer as measured zero")
        return (row.values - self.mean) / self.scale


def preview_state(state: Array, telemetry: Array | None = None, *, max_state_dim: int = 32) -> Array:
    """Offline feature-width check only; does not modify a model or input mapping."""
    state = np.asarray(state)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError("Expected finite baseline 16D state")
    if telemetry is not None and (telemetry.shape != (32,) or not np.isfinite(telemetry).all()):
        raise ValueError("Expected finite velocity/effort features")
    result = state.copy() if telemetry is None else np.concatenate((state, telemetry))
    if len(result) > max_state_dim:
        raise ValueError("SmolVLA state projection must explicitly support the feature width")
    return result
