"""Hydrogen link-atom (cap) primitives shared across the codebase.

Caps are virtual hydrogen atoms placed on the QM/ML side of a Q-M
boundary bond. Their positions are deterministic functions of the
Q (ML-side) and M (MM-side) atom positions; their forces from the
ML evaluator must be redistributed onto Q and M before being handed
back to OpenMM.

Two callers:

- `openmmml/models/macepotential.py::_computeMACE` — places caps in
  MACE's input each step and redistributes forces from MACE's output.
- `openmmml/mlpotential.py::_build_oniom_mixed_system` — places caps
  in the ONIOM low-model `Context` each step and redistributes forces
  from the model `Context`'s output.

Both call sites need the *same* placement formula and the *same*
redistribution Jacobian, so the math lives here and both import it.
The math:

    r_cap = (1 - C_L) * r_Q + C_L * r_M       where  C_L = target_dist / |r_M - r_Q|

    F_Q  ← F_Q + (1 - C_L) F_cap + C_L (F_cap · ê_b) ê_b
    F_M  ← F_M + C_L F_cap       - C_L (F_cap · ê_b) ê_b
    where ê_b = (r_M - r_Q) / |r_M - r_Q|

The redistribution preserves total force and total torque (the cap is
constrained to lie along the Q-M axis at fixed fraction `C_L`, so
its motion is entirely determined by Q and M motion; the chain rule
of energy w.r.t. Q,M gives the formulas above).
"""
from __future__ import annotations

import numpy as np


def compute_cap_positions(
    r_Q: np.ndarray,
    r_M: np.ndarray,
    target_dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cap positions and the per-cap C_L ratio.

    Parameters
    ----------
    r_Q : array of shape (K, 3)
        Position of each Q atom (ML side of the boundary bond), one
        row per cap, in the same length unit as `target_dist`.
    r_M : array of shape (K, 3)
        Position of each M atom (MM side), aligned 1:1 with `r_Q`.
    target_dist : array of shape (K,)
        Target distance from Q to the cap. Typically a hydrogen bond
        length (e.g., 1.09 Å for C-H).

    Returns
    -------
    r_cap : array of shape (K, 3)
        Cap positions.
    C_L : array of shape (K,)
        The placement ratio used for each cap. The caller usually
        also needs `C_L` for the corresponding force redistribution.

    Raises
    ------
    ValueError
        If any C_L falls outside (0, 1) — meaning the requested
        target distance exceeds (or equals) the actual Q-M distance,
        or Q and M are coincident. Both cases are user-error.
    """
    r_Q = np.asarray(r_Q, dtype=np.float64)
    r_M = np.asarray(r_M, dtype=np.float64)
    target_dist = np.asarray(target_dist, dtype=np.float64)
    v = r_M - r_Q
    s = np.linalg.norm(v, axis=-1)
    if np.any(s == 0.0):
        raise ValueError("Cap Q and M atoms are coincident.")
    C_L = target_dist / s
    if np.any(~np.isfinite(C_L)) or np.any(C_L <= 0.0) or np.any(C_L >= 1.0):
        raise ValueError(
            f"Link-atom C_L out of (0, 1): {C_L}. "
            "Check that target_dist < |r_M - r_Q| for every cap."
        )
    r_cap = (1.0 - C_L)[:, None] * r_Q + C_L[:, None] * r_M
    return r_cap, C_L


def redistribute_cap_force(
    F_cap: np.ndarray,
    r_Q: np.ndarray,
    r_M: np.ndarray,
    C_L: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map per-cap forces onto the corresponding Q and M atoms.

    Parameters
    ----------
    F_cap : array of shape (K, 3)
        Force on each cap from the ML evaluator.
    r_Q, r_M : arrays of shape (K, 3)
        Positions of the Q (ML side) and M (MM side) atoms.
    C_L : array of shape (K,)
        Placement ratio for each cap (same as returned by
        `compute_cap_positions`).

    Returns
    -------
    F_Q_add : array of shape (K, 3)
        Force contribution to add onto each Q atom.
    F_M_add : array of shape (K, 3)
        Force contribution to add onto each M atom.

    Notes
    -----
    The caller is responsible for accumulating the returned
    contributions onto the right global atom indices (e.g., via
    `f[q_global] += F_Q_add` if multiple caps share a Q or M atom
    is forbidden by `_prepareLinkRecords`).
    """
    F_cap = np.asarray(F_cap, dtype=np.float64)
    r_Q = np.asarray(r_Q, dtype=np.float64)
    r_M = np.asarray(r_M, dtype=np.float64)
    C_L = np.asarray(C_L, dtype=np.float64)
    v = r_M - r_Q
    s = np.linalg.norm(v, axis=-1)
    e_b = v / s[:, None]
    proj = np.einsum("ki,ki->k", F_cap, e_b)
    F_Q_add = (1.0 - C_L)[:, None] * F_cap + (C_L * proj)[:, None] * e_b
    F_M_add = C_L[:, None] * F_cap - (C_L * proj)[:, None] * e_b
    return F_Q_add, F_M_add
