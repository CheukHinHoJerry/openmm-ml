"""Internal helpers shared across openmm-ml embedding modes.

Currently exposes link-atom (cap) primitives used by both the MACE
PythonForce in `openmmml/models/macepotential.py` and the ONIOM-EE
correction PythonForce in `openmmml/mlpotential.py`.
"""
from openmmml.embedding._links import (
    compute_cap_positions,
    redistribute_cap_force,
)

__all__ = ["compute_cap_positions", "redistribute_cap_force"]
