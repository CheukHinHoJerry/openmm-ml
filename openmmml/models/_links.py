"""Hydrogen link-atom (cap) primitives shared across the codebase.

Caps are virtual hydrogen atoms placed on the QM/ML side of a Q-M
boundary bond. Their positions are deterministic functions of the
Q (ML-side) and M (MM-side) atom positions; their forces from the
ML evaluator must be redistributed onto Q and M before being handed
back to OpenMM.

The caller is `openmmml/models/macepotential.py::_computeMACE`, which
places caps in MACE's input each step and redistributes forces from
MACE's output.  The math lives here rather than there so that any
further evaluator needing caps uses the *same* placement formula and
the *same* redistribution Jacobian:

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


# All 27 neighbouring lattice-cell offsets (coefficients in {-1, 0, 1}^3).
_OFFSETS_27 = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)


def minimum_image_M(
    r_Q: np.ndarray,
    r_M: np.ndarray,
    cell: np.ndarray,
    max_bond: float = 2.5,
) -> np.ndarray:
    """Return each M atom shifted to its minimum image relative to its Q partner.

    When a Q-M boundary bond straddles a periodic boundary the raw M position is
    in a different image than Q, and both cap *placement* and cap-force
    *redistribution* must use the same imaged M (otherwise the redistributed
    force is not the gradient of the energy the cap produced). Both call sites in
    ``macepotential._computeMACE`` therefore route ``r_M`` through this helper.

    The nearest image is found by a 27-cell search: round the fractional
    displacement to the nearest lattice cell, then pick the shortest candidate
    over that cell and its 26 neighbours. This is exact for a real (weakly skewed)
    simulation box; it is *not* a certified closest-vector solver for
    pathologically skewed triclinic cells, which do not occur for physical MD
    systems. As a sanity check, the imaged Q-M distance must stay below
    ``max_bond`` (Angstrom); a longer bond means a genuinely wrapped / broken pair
    (or a box thinner than the bond) and raises rather than placing a bad cap.

    Parameters
    ----------
    r_Q, r_M : arrays of shape (K, 3)
        Q (ML-side) and M (MM-side) positions, one row per cap, in Angstrom
        (the unit ``cell`` is given in).
    cell : array of shape (3, 3)
        Periodic box vectors as rows (OpenMM convention), in Angstrom.
    max_bond : float, optional
        Chemical sanity ceiling on the imaged Q-M bond length (Angstrom). Frontier
        bonds are ~1.0-1.6 A; the default 2.5 A leaves margin while still catching
        a wrongly-imaged / wrapped pair.

    Returns
    -------
    r_M_imaged : array of shape (K, 3)
        M positions shifted to the image nearest their Q partner.
    """
    r_Q = np.atleast_2d(np.asarray(r_Q, dtype=np.float64))
    r_M = np.atleast_2d(np.asarray(r_M, dtype=np.float64))
    cell = np.asarray(cell, dtype=np.float64)

    if cell.shape != (3, 3):
        raise ValueError("cell must have shape (3, 3)")
    if abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("cell is singular or nearly singular")
    if r_Q.shape != r_M.shape:
        raise ValueError("r_Q and r_M must have the same shape")

    dr = r_M - r_Q
    base = np.round(dr @ np.linalg.inv(cell))          # nearest lattice cell (orthorhombic guess)
    shifts = base[:, None, :] + _OFFSETS_27[None, :, :]  # (K, 27, 3) integer coefficients
    cand = dr[:, None, :] - shifts @ cell              # (K, 27, 3) candidate displacements
    best = np.argmin(np.einsum("kij,kij->ki", cand, cand), axis=1)
    dr_mi = cand[np.arange(cand.shape[0]), best]

    bond = np.linalg.norm(dr_mi, axis=-1)
    if np.any(bond >= max_bond):
        raise ValueError(
            f"imaged Q-M link bond length {bond.max():.2f} A exceeds max_bond "
            f"{max_bond:.2f} A; the pair is genuinely wrapped/broken (or the box is "
            "thinner than the bond). Increase max_bond only if this bond is real."
        )
    return r_Q + dr_mi


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


# -----------------------------------------------------------------------------
# Link-atom CHARGE redistribution (Z1 / DZ1)
#
# Standard QM/MM correction. When a Q–M bond is cut and the M atom is replaced
# (on the QM side) by an H link atom, the M atom's full partial charge sits
# ~1.5 Å from the nearest QM atom. With electrostatic embedding the QM region
# sees that close-in partial charge through the link H and gets
# over-polarised. The "charge shift" / "redistributed charge" schemes move
# the M partial charge away from the boundary.
#
# Z1   — set q_M ← 0. Cheapest; breaks total MM-charge neutrality by -q_M_orig.
# DZ1  — Z1 + redistribute q_M_orig / N(M1) onto each MM neighbour (M1 atom).
#        Preserves total MM charge to round-off.
#
# Both schemes only modify the *constant* mm_charges array fed into the ML
# potential's electrostatic embedding — they do not touch the OpenMM MM
# scaffold's NonbondedForce, which keeps the MM-MM Coulomb sum bit-exact
# with the original force field (matching standard QM/MM practice).
# -----------------------------------------------------------------------------

def apply_link_charge_redistribution(
    mm_atoms: np.ndarray,
    mm_charges: np.ndarray,
    link_info: dict,
    topology,
    scheme: str,
) -> np.ndarray:
    """Apply Z1 / DZ1 link-atom charge redistribution to an MM charges array.

    Parameters
    ----------
    mm_atoms : np.ndarray, shape (n_mm,)
        Global atom indices of the MM atoms, in the order their charges appear
        in ``mm_charges``.
    mm_charges : np.ndarray, shape (n_mm,)
        Original MM partial charges aligned with ``mm_atoms``.
    link_info : dict
        Output of ``MACEPotentialImpl._prepareLinkRecords``; must contain
        ``q_global`` and ``m_global`` arrays.
    topology : openmm.app.Topology
        The full system topology — needed to look up M's MM neighbours (M1) for
        DZ1.
    scheme : str
        One of ``"none"``, ``"z1"``, ``"dz1"``.

    Returns
    -------
    np.ndarray, shape (n_mm,)
        New MM charges array (always a copy of the input; never an alias).

    Notes
    -----
    For ``"dz1"``, if any M atom has zero MM neighbours the call falls back to
    Z1 for that atom (charge zeroed, nothing to redistribute) and emits a
    ``UserWarning``. Total MM charge will then drift by the orphan q_M_orig.
    """
    scheme = scheme.lower()
    if scheme not in {"none", "z1", "dz1"}:
        raise ValueError(
            f"Unsupported linkChargeScheme {scheme!r}. "
            "Supported in this iteration: 'none', 'z1', 'dz1'."
        )
    new_charges = np.asarray(mm_charges, dtype=np.float64).copy()
    if scheme == "none":
        return new_charges

    mm_idx_to_row = {int(g): k for k, g in enumerate(mm_atoms)}
    qm_set = set(int(g) for g in link_info["q_global"])
    m_globals = [int(g) for g in link_info["m_global"]]

    # Snapshot original q_M before any zeroing so DZ1 redistributes the
    # *original* value even if two M atoms happened to be the same row (they
    # are validated unique upstream, but the snapshot is also free insurance).
    orig_qM = {m: float(new_charges[mm_idx_to_row[m]]) for m in m_globals}

    if scheme == "z1":
        for m in m_globals:
            new_charges[mm_idx_to_row[m]] = 0.0
        return new_charges

    # ---- scheme == "dz1" ----
    # Build the MM-neighbour list for each M atom by walking topology bonds.
    # Filter: the neighbour must (a) be in mm_atoms, (b) not be a Q atom.
    # The QM filter is redundant against (a) when q_global ⊂ atoms ⊂ mm complement,
    # but cheap and explicit.
    m_set = set(m_globals)
    mm_neighbors_of_M: dict[int, list[int]] = {m: [] for m in m_globals}
    for bond in topology.bonds():
        a = bond.atom1.index
        b = bond.atom2.index
        if a in m_set and b in mm_idx_to_row and b not in qm_set:
            mm_neighbors_of_M[a].append(b)
        if b in m_set and a in mm_idx_to_row and a not in qm_set:
            mm_neighbors_of_M[b].append(a)

    import warnings
    for m in m_globals:
        new_charges[mm_idx_to_row[m]] = 0.0
        m1s = mm_neighbors_of_M[m]
        if not m1s:
            warnings.warn(
                f"DZ1: M atom {m} has no MM neighbours to redistribute "
                f"q={orig_qM[m]:+.4f} e onto. Falling back to Z1 for this atom.",
                stacklevel=2,
            )
            continue
        share = orig_qM[m] / len(m1s)
        for m1 in m1s:
            new_charges[mm_idx_to_row[m1]] += share

    # Conservation check: warn if numerically non-trivial drift (DZ1 should
    # preserve total to f64 round-off).
    delta = float(new_charges.sum()
                  - np.asarray(mm_charges, dtype=np.float64).sum())
    if abs(delta) > 1e-9:
        import warnings as _w
        _w.warn(
            f"DZ1: post-redistribution total MM charge drift = {delta:+.3e} e "
            "(likely due to M atoms with zero MM neighbours).",
            stacklevel=2,
        )
    return new_charges
