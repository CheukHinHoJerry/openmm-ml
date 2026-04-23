"""
macepotential.py: Implements the MACE potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr, Joao Morado

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from typing import Iterable, Optional, Sequence, Tuple, Union
from functools import partial
from pathlib import Path
import numpy as np

LinkRecordTuple = Tuple[int, int, float]  # (q_global, m_global, target_dist)
LinkRecordsArg = Union[str, "os.PathLike[str]", Sequence[LinkRecordTuple], None]


class MACEPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MACEPotentialImpl objects."""

    def createImpl(
        self, name: str, modelPath: Optional[str] = None, **args
    ) -> MLPotentialImpl:
        return MACEPotentialImpl(name, modelPath)


class MACEPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the MACE potential.

    The MACE potential is constructed using MACE to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.
    This implementation supports both foundation models and locally trained MACE models.

    To use one of the pre-trained MACE foundation models, specify the model name. For example:

    >>> potential = MLPotential('mace-off23-small')

    Other available models include 'mace-off23-medium', 'mace-off23-large', 'mace-off24-medium',
    'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium', and 'mace-omol-0-extra-large'.

    To use a locally trained MACE model, provide the path to the model file. For example:

    >>> potential = MLPotential('mace', modelPath='MACE.model')

    During system creation, you can optionally specify the precision of the model using the
    ``precision`` keyword argument. Supported options are 'single' and 'double'. For example:

    >>> system = potential.createSystem(topology, precision='single')

    By default, the implementation uses the precision of the loaded MACE model.
    According to the MACE documentation, 'single' precision is recommended for MD (faster but
    less accurate), while 'double' precision is recommended for geometry optimization.

    Additionally, you can request computation of the full atomic energy, including the atom
    self-energy, instead of the default interaction energy, by setting ``returnEnergyType`` to
    'energy'. For example:
    
    >>> system = potential.createSystem(topology, returnEnergyType='energy')

    The default is to compute the interaction energy, which can be made explicit by setting
    ``returnEnergyType='interaction_energy'``.

    Attributes
    ----------
    name : str
        The name of the MACE model.
    modelPath : str
        The path to the locally trained MACE model if ``name`` is 'mace'.
    """

    def __init__(self, name: str, modelPath) -> None:
        """
        Initialize the MACEPotentialImpl.

        Parameters
        ----------
        name : str
            The name of the MACE model.
            Options include 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large',
            'mace-off24-medium', 'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium',
            'mace-omol-0-extra-large', and 'mace'.
        modelPath : str, optional
            The path to the locally trained MACE model if ``name`` is 'mace'.
        """
        self.name = name
        self.modelPath = modelPath

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        precision: Optional[str] = None,
        returnEnergyType: str = "interaction_energy",
        linkRecords: LinkRecordsArg = None,
        **args,
    ) -> None:
        """
        Add the MACEForce to the OpenMM System.

        Parameters
        ----------
        topology : openmm.app.Topology
            The topology of the system.
        system : openmm.System
            The system to which the force will be added.
        atoms : iterable of int
            The indices of the atoms to include in the model. If ``None``, all atoms are included.
        forceGroup : int
            The force group to which the force should be assigned.
        precision : str, optional
            The precision of the model. Supported options are 'single' and 'double'.
            If ``None``, the default precision of the model is used.
        returnEnergyType : str, optional
            Whether to return the interaction energy or the energy including the self-energy.
            Default is 'interaction_energy'. Supported options are 'interaction_energy' and 'energy'.
        linkRecords : str / path / sequence of (q_global, m_global, target_dist) / None
            Hydrogen link-atom cap records for QM/MM boundary bonds. Either a
            path to a capping_mapping.csv emitted by cli/cap_qm_boundary.py, or
            a sequence of ``(q_global, m_global, target_dist_ang)`` tuples
            where indices are into the full OpenMM system (0-based).
            ``q_global`` must be in ``atoms``; ``m_global`` must not be in
            ``atoms``. Each cap adds one fictitious H node to the ML model's
            input and redistributes its force back onto (Q, M) via the
            distance-mode chain rule. Non-periodic systems only.
            Default ``None`` preserves current behavior byte-for-byte.
            Geometry + force redistribution only — MM point-charge
            redistribution on M is *not* performed; see
            docs/plans/link-atom-inference.md.
        """
        import torch
        try:
            from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
            from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol
        except ImportError as e:
            raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")
        try:
            from e3nn.util import jit
        except ImportError as e:
            raise ImportError(f"Failed to import e3nn with error: {e}. Install e3nn with 'pip install e3nn'.")

        assert returnEnergyType in ["interaction_energy", "energy"], f"Unsupported returnEnergyType: '{returnEnergyType}'. Supported options are 'interaction_energy' or 'energy'."

        # Load the model.

        models = {
            'mace-off23-small': (mace_off, 'small', True),
            'mace-off23-medium': (mace_off, 'medium', True),
            'mace-off23-large': (mace_off, 'large', True),
            'mace-off24-medium': (mace_off, 'https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true', True),
            'mace-mpa-0-medium': (mace_mp, 'medium-mpa-0', False),
            'mace-omat-0-small': (mace_mp, 'small-omat-0', True),
            'mace-omat-0-medium': (mace_mp, 'medium-omat-0', True),
            'mace-omol-0-extra-large': (mace_omol, 'extra_large', True)
        }
        device = self._getTorchDevice(args)
        print(f"====== Running MACE potential on device: {device} =======")
        if self.name in models:
            fn, name, warn = models[self.name]
            model = fn(model=name, device=device, return_raw_model=True).to(device)
            if warn:
                import logging
                logging.warning(f'The model {self.name} is distributed under the restrictive ASL license.  Commercial use is not permitted.')
        elif self.name == "mace":
            if self.modelPath is not None:
                model = torch.load(self.modelPath, map_location=device)
                if hasattr(model, "to"):
                    model = model.to(device)
            else:
                raise ValueError("No modelPath provided for local MACE model.")
        else:
            raise ValueError(f"Unsupported MACE model: {self.name}")

        # Get the atomic numbers of the ML region.

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomicNumbers = [atom.element.atomic_number for atom in includedAtoms]

        # Link-atom capping on QM/MM boundary bonds (optional).
        # Parsed once at construction; the set of caps is static for the run.
        # Each cap appends a fictitious H node to the model input (nodeAttrs,
        # ptr, batch) and has its force redistributed back onto (Q, M) inside
        # _computeMACE. See docs/plans/link-atom-inference.md.
        linkInfo = _prepareLinkRecords(linkRecords, atoms, topology, system)
        if linkInfo is not None:
            atomicNumbers = atomicNumbers + [1] * linkInfo["K"]

        # Set the precision that the model will be used with.

        modelDefaultDtype = next(model.parameters()).dtype
        if precision is None:
            dtype = modelDefaultDtype
        elif precision == "single":
            dtype = torch.float32
        elif precision == "double":
            dtype = torch.float64
        else:
            raise ValueError(f"Unsupported precision {precision} for the model. Supported values are 'single' and 'double'.")
        if dtype != modelDefaultDtype:
            print(f"Model dtype is {modelDefaultDtype} and requested dtype is {dtype}. The model will be converted to the requested dtype.")

        # One hot encoding of atomic numbers

        model_device = device
        try:
            model_device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            pass

        zTable = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        nodeAttrs = to_one_hot(
            torch.tensor(atomic_numbers_to_indices(atomicNumbers, z_table=zTable), dtype=torch.long, device=model_device).unsqueeze(-1),
            num_classes=len(zTable))

        if atoms is None:
            indices = None
        else:
            indices = np.array(atoms)
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

        # Create the PythonForce and add it to the System.

        compute = partial(_computeMACE,
                          model=model,
                          ptr=torch.tensor([0, nodeAttrs.shape[0]], dtype=torch.long, device=model_device, requires_grad=False),
                          node_attrs=nodeAttrs.to(dtype),
                          batch=torch.zeros(nodeAttrs.shape[0], dtype=torch.long, device=model_device, requires_grad=False),
                          pbc=torch.tensor([periodic, periodic, periodic], dtype=torch.bool, device=model_device, requires_grad=False),
                          returnEnergyType=returnEnergyType,
                          charge=torch.tensor([float(args.get('charge', 0))], dtype=dtype, device=model_device, requires_grad=False),
                          multiplicity=torch.tensor([float(args.get('multiplicity', 1))], dtype=dtype, device=model_device, requires_grad=False),
                          indices=indices,
                          periodic=periodic,
                          linkInfo=linkInfo)
        PythonForce = getattr(openmm, "PythonForce", None)
        if PythonForce is None:
            PythonForce = getattr(getattr(openmm, "openmm", None), "PythonForce", None)
        if PythonForce is None:
            raise RuntimeError("PythonForce is not available in this OpenMM build.")
        force = PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


def _computeMACE(state, model, ptr, node_attrs, batch, pbc, returnEnergyType, charge, multiplicity, indices, periodic, linkInfo=None):
    import os
    import torch
    from mace.data.neighborhood import get_neighborhood
    energyScale = 96.4853
    lengthScale = 10.0
    positions_full = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    numAtoms = positions_full.shape[0]
    if indices is not None:
        positions = positions_full[indices]
    else:
        positions = positions_full

    # Link atoms: append K fictitious H positions placed each step from current
    # Q, M coordinates. These extend the model input only; they never enter the
    # OpenMM system. See docs/plans/link-atom-inference.md.
    if linkInfo is not None:
        if indices is None:
            raise ValueError("linkRecords requires an explicit `atoms` subset.")
        r_Q = positions_full[linkInfo["q_global"]]
        r_M = positions_full[linkInfo["m_global"]]
        v = r_M - r_Q
        s = np.linalg.norm(v, axis=1)
        if np.any(s == 0.0):
            raise ValueError("Link-atom Q and M coincident at this step.")
        C_L = linkInfo["target_dist"] / s
        pos_link = (1.0 - C_L)[:, None] * r_Q + C_L[:, None] * r_M
        positions = np.concatenate([positions, pos_link], axis=0)

    if periodic:
        cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    else:
        cell = np.identity(3, dtype=np.float64)
    dtype = node_attrs.dtype
    cutoff = float(model.r_max.detach())
    edgeIndex, shifts, _, _ = get_neighborhood(positions, cutoff, [periodic, periodic, periodic], cell)
    cell_tensor = torch.tensor(cell, dtype=dtype, device=ptr.device)
    volume = torch.linalg.det(cell_tensor)
    if torch.abs(volume) > 0:
        rcell = 2 * torch.pi * torch.linalg.inv(cell_tensor.mT)
    else:
        rcell = torch.zeros((3, 3), dtype=dtype, device=ptr.device)
    inputDict = {
        "ptr": ptr,
        "node_attrs": node_attrs,
        "batch": batch,
        "pbc": pbc,
        "positions": torch.tensor(positions, dtype=dtype, device=ptr.device),
        "edge_index": torch.tensor(edgeIndex, dtype=torch.int64, device=ptr.device),
        "shifts": torch.tensor(shifts, dtype=dtype, device=ptr.device),
        "cell": cell_tensor,
        "rcell": rcell,
        "volume": volume.reshape(-1),
        "total_charge": charge,
        "total_spin": multiplicity,
        "external_field": torch.zeros((charge.shape[0], 3), dtype=dtype, device=ptr.device),
        "fermi_level": torch.zeros((1,), dtype=dtype, device=ptr.device)
    }
    results = model(inputDict, compute_force=True)
    energy = float(results[returnEnergyType].detach())*energyScale
    forces = (results["forces"]*energyScale*lengthScale).detach().cpu().numpy()
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=(np.float64 if dtype == torch.float64 else np.float32))
        if linkInfo is None:
            f[indices] = forces
        else:
            N = len(indices)
            f_ml = forces[:N]
            f_link = forces[N:]
            f[indices] = f_ml
            # Redistribute each link atom's force onto its (Q, M) partners.
            # See _prepareLinkRecords for the bookkeeping. r_Q, r_M, s, C_L
            # are recomputed here from positions_full (cheap; K is small).
            r_Q = positions_full[linkInfo["q_global"]]
            r_M = positions_full[linkInfo["m_global"]]
            v = r_M - r_Q
            s = np.linalg.norm(v, axis=1)
            C_L = linkInfo["target_dist"] / s
            e_b = v / s[:, None]
            proj = np.einsum("ki,ki->k", f_link, e_b)
            F_Q_add = (1.0 - C_L)[:, None] * f_link + (C_L * proj)[:, None] * e_b
            F_M_add = C_L[:, None] * f_link - (C_L * proj)[:, None] * e_b
            # q_global and m_global were validated unique at construction, so
            # plain += is correct (no duplicate-row aggregation needed).
            f[linkInfo["q_global"]] += F_Q_add.astype(f.dtype, copy=False)
            f[linkInfo["m_global"]] += F_M_add.astype(f.dtype, copy=False)
        forces = f
    return energy, forces


def _prepareLinkRecords(linkRecords, atoms, topology, system):
    """Normalize the ``linkRecords`` argument into a frozen bundle for the
    per-step closure. Returns ``None`` if no link records were supplied.

    Assertions (failing loudly):
      * non-periodic system (PBC deferred to a follow-up PR)
      * ``atoms`` is not None
      * every ``q_global`` is in ``atoms``
      * no ``m_global`` is in ``atoms``
      * every (Q, M) pair is unique, and no atom appears as Q or M in more
        than one cap (simplifies per-step scatter; relax later if needed)
    """
    if linkRecords is None:
        return None

    # Non-periodic only (locked assumption, see docs/plans/link-atom-inference.md).
    is_periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
    if is_periodic:
        raise ValueError(
            "linkRecords is only supported for non-periodic systems in this "
            "implementation; PBC min-image placement is deferred."
        )
    if atoms is None:
        raise ValueError("linkRecords requires an explicit `atoms` subset.")

    if isinstance(linkRecords, (str, Path)):
        # Defer import so openmm-ml doesn't require mlmm for the default path.
        from mlmm.link_atoms import load_link_records
        recs = load_link_records(linkRecords)
        # capping_mapping.csv is 1-based; convert to 0-based OpenMM indices.
        tuples = [(r.q_idx1 - 1, r.m_idx1 - 1, r.target_dist) for r in recs]
    else:
        tuples = [(int(q), int(m), float(td)) for (q, m, td) in linkRecords]

    if not tuples:
        return None

    atoms_set = set(int(a) for a in atoms)
    seen_pairs: set = set()
    seen_q: set = set()
    seen_m: set = set()
    q_global = np.empty(len(tuples), dtype=np.int64)
    m_global = np.empty(len(tuples), dtype=np.int64)
    target_dist = np.empty(len(tuples), dtype=np.float64)
    for k, (q, m, td) in enumerate(tuples):
        if q not in atoms_set:
            raise ValueError(f"linkRecords[{k}]: q_global={q} is not in `atoms`.")
        if m in atoms_set:
            raise ValueError(f"linkRecords[{k}]: m_global={m} is in `atoms` (should be MM).")
        if (q, m) in seen_pairs:
            raise ValueError(f"linkRecords[{k}]: duplicate (Q, M) pair ({q}, {m}).")
        if q in seen_q or m in seen_m:
            raise ValueError(
                f"linkRecords[{k}]: atom {q if q in seen_q else m} appears in more than "
                "one cap; current implementation requires unique Q and M across caps."
            )
        if td <= 0:
            raise ValueError(f"linkRecords[{k}]: target_dist must be positive; got {td}.")
        seen_pairs.add((q, m))
        seen_q.add(q)
        seen_m.add(m)
        q_global[k] = q
        m_global[k] = m
        target_dist[k] = td
    return {
        "K": len(tuples),
        "q_global": q_global,
        "m_global": m_global,
        "target_dist": target_dist,
    }