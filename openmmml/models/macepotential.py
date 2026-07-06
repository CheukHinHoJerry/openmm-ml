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

    By default the reported energy is the full ``energy`` returned by the MACE
    model — the same scalar whose gradient w.r.t. positions is reported as the
    force, so the resulting potential is exactly conservative. To get only the
    message-passing readout component, set ``returnEnergyType='interaction_energy'``:

    >>> system = potential.createSystem(topology, returnEnergyType='interaction_energy')

    Note: ``returnEnergyType='interaction_energy'`` is **not** energy/force
    consistent for the PolarMACE family, which adds Coulomb / dipole / local-
    electron terms to ``total_energy`` whose gradients are in ``forces`` but
    which are not in ``interaction_energy``. Using it produces a non-zero,
    delta-independent floor in any finite-difference force check and
    apparent NVE drift in MD.

    Precision caveat for ``returnEnergyType='energy'``: this key returns the
    full ``total_energy = e0 + inter_e + extras`` where ``e0`` are the
    model's per-atom reference energies. For foundation models (mace-mp,
    mace-off, mace-omat, ...) ``e0`` is typically tens of eV per atom, so the
    reported scalar for a large ML region can be 10⁴–10⁶ eV in magnitude.
    The **forces** stay exact at any scale (they are gradients of this same
    scalar), but the **energy** column written into single-precision OpenMM
    state files / log lines may carry only ~6–7 significant digits at that
    magnitude — meV resolution is lost. Use ``precision='double'`` if you
    need accurate absolute energies, or note that energy differences (e.g.
    NVE drift) still resolve cleanly because the e0 contribution cancels in
    the difference. A runtime warning fires when this regime is detected.

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
        returnEnergyType: str = "energy",
        linkRecords: LinkRecordsArg = None,
        linkChargeScheme: str = "dz1",
        embedding: str = "mechanical",
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
            Which scalar from the MACE model output is reported to OpenMM as
            the potential energy. Default ``'energy'`` is the same quantity
            the force vector is differentiated against, so OpenMM sees a
            self-consistent (conservative) potential. ``'interaction_energy'``
            returns only the message-passing readout; for PolarMACE this is
            **not** the gradient partner of ``forces`` and will produce
            apparent NVE drift / a non-zero finite-difference plateau.
        linkRecords : str / path / sequence of (q_global, m_global, target_dist) / None
            Hydrogen link-atom cap records for QM/MM boundary bonds.
        linkChargeScheme : str, optional
            How to handle the partial charges on the MM-side boundary atoms
            (M atoms) when ``linkRecords`` is provided. The M-atom partial
            charge would otherwise sit ~1.5 Å from the nearest QM atom (through
            the link H) and over-polarise the QM region's MACE-predicted
            electronic structure. Supported in this iteration:

            - ``"none"``: leave MM charges untouched (legacy behaviour).
            - ``"z1"``: set q_M = 0 for every M atom. Cheapest fix; breaks
              total MM-charge neutrality by -q_M_orig.
            - ``"dz1"`` (default): q_M = 0 plus q_M_orig is distributed
              equally onto M's MM neighbours (M1 atoms). Preserves total
              MM charge to round-off. Falls back to Z1 (with a warning) for
              any M atom that has zero MM neighbours.

            Only modifies the MM charge array passed to the ML potential;
            the OpenMM ``NonbondedForce`` is left untouched, so MM-MM Coulomb
            is bit-exact with the original force field (standard QM/MM
            practice). Z2 and RCD (which require per-step virtual charges)
            are not in this iteration.
        embedding : {"mechanical", "electrostatic"}
            Mixed-system embedding mode for local PolarMACE models. ``mechanical``
            preserves the previous behavior: MM positions/charges are not passed
            into MACE. ``electrostatic`` passes MM positions/charges into
            PolarMACE, removes classical ML-MM Coulomb from the MM
            ``NonbondedForce``, and scatters returned ``mm_forces`` back onto
            MM atoms. Non-PolarMACE models ignore this option and behave as
            mechanical embedding.
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

        use_mm_embedding = _should_use_mm_embedding(model, atoms, embedding)

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomicNumbers = [atom.element.atomic_number for atom in includedAtoms]

        # Precision warning for returnEnergyType="energy".
        # The 'energy' key is total_energy = e0 + inter_e + (PolarMACE extras).
        # For models with non-trivial per-atom reference energies (mace-mp,
        # mace-off, mace-omat foundations), e0 dominates the reported scalar
        # — absolute values can reach 10^4–10^6 eV for large systems. That
        # absolute scale is what gets written into single-precision OpenMM
        # state/log lines, so the energy column may carry as few as 6–7
        # significant digits and lose resolution at the meV level even though
        # the *forces* (gradients of total_energy) remain accurate.
        # Conservation/drift diagnostics still work (they're differences),
        # but absolute-energy comparisons across runs need the model's e0
        # baseline subtracted, or a double-precision report.
        if returnEnergyType == "energy":
            try:
                e0 = model.atomic_energies_fn.atomic_energies
                e0_max = float(e0.detach().abs().max())
            except AttributeError:
                e0_max = 0.0
            if e0_max > 1.0:  # 1 eV per atom is conservative; foundation models far exceed this
                import warnings as _w
                _w.warn(
                    f"returnEnergyType='energy' includes per-atom reference "
                    f"energies (max |e0| = {e0_max:.2f} eV/atom over {int(e0.numel())} "
                    f"element entries). For a {len(includedAtoms)}-atom ML region the "
                    f"absolute reported energy scale can reach ~{e0_max * len(includedAtoms):.0f} "
                    f"eV; single-precision floats will lose meV-level resolution at that "
                    f"magnitude. Use precision='double' if you need accurate absolute "
                    f"energies, or pass returnEnergyType='interaction_energy' for the "
                    f"e0-subtracted readout (note: only 'energy' is gradient-consistent "
                    f"with the reported forces for PolarMACE — see the docstring).",
                    stacklevel=2,
                )

        linkInfo = _prepareLinkRecords(linkRecords, atoms, topology, system)
        if linkInfo is not None:
            atomicNumbers = atomicNumbers + [1] * linkInfo["K"]

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
            # Actually do the conversion. The previous code only printed the
            # warning and left the model untouched, which caused dtype
            # mismatches inside e3nn's compiled TensorProduct submodules
            # when inputs were passed at the requested dtype.
            model = model.to(dtype)

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
        mmInfo = None
        if use_mm_embedding:
            # ML-MM Coulomb is removed by MLPotential.createMixedSystem when
            # embedding='electrostatic'. We only need MM positions/charges for
            # the PolarMACE input here.
            mmInfo = _prepareMMEmbedding(system, atoms)

            # Optional Z1 / DZ1 link-atom charge redistribution. Standard QM/MM
            # correction to stop the QM region from being over-polarised by
            # the partial charge on the MM-side boundary atom (M atom) sitting
            # ~1.5 Å from the link H. Only modifies the *constant* mm_charges
            # array baked into the PythonForce closure; does not touch the
            # OpenMM NonbondedForce, so MM-MM Coulomb stays bit-exact with the
            # original FF (standard practice).
            if linkInfo is not None and linkChargeScheme not in (None, "none"):
                from openmmml.embedding._links import (
                    apply_link_charge_redistribution as _apply_link_q,
                )
                mmInfo["mm_charges"] = _apply_link_q(
                    mm_atoms=mmInfo["mm_atoms"],
                    mm_charges=mmInfo["mm_charges"],
                    link_info=linkInfo,
                    topology=topology,
                    scheme=linkChargeScheme,
                )
                print(f"[link-charge-redistribution] scheme={linkChargeScheme}  "
                      f"M atoms touched={len(linkInfo['m_global'])}")
        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()

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
                          linkInfo=linkInfo,
                          mmInfo=mmInfo)
        PythonForce = getattr(openmm, "PythonForce", None)
        if PythonForce is None:
            PythonForce = getattr(getattr(openmm, "openmm", None), "PythonForce", None)
        if PythonForce is None:
            raise RuntimeError("PythonForce is not available in this OpenMM build.")
        force = PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


def _supports_mm_embedding(model) -> bool:
    return model.__class__.__name__ == "PolarMACE"


_SUPPORTED_EMBEDDINGS = ("mechanical", "electrostatic", "oniom-electrostatic")
_MM_EMBEDDING_MODES = ("electrostatic", "oniom-electrostatic")


def _should_use_mm_embedding(model, atoms: Optional[Iterable[int]], embedding: str) -> bool:
    if embedding not in _SUPPORTED_EMBEDDINGS:
        raise ValueError(
            f"Unsupported embedding mode '{embedding}'. Supported values are "
            + ", ".join(repr(m) for m in _SUPPORTED_EMBEDDINGS)
            + "."
        )
    return (
        _supports_mm_embedding(model)
        and atoms is not None
        and embedding in _MM_EMBEDDING_MODES
    )


def _prepareMMEmbedding(system: openmm.System, atoms: Optional[Iterable[int]]):
    """Extract the MM complement and its charges from the system's NonbondedForce."""
    if atoms is None:
        return None

    num_particles = int(system.getNumParticles())
    ml_atoms = np.asarray(list(atoms), dtype=np.int64)
    ml_set = set(int(i) for i in ml_atoms.tolist())
    mm_atoms = np.asarray(
        [i for i in range(num_particles) if i not in ml_set], dtype=np.int64
    )

    nonbonded = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise ValueError(
            "PolarMACE MM embedding requires a NonbondedForce to source MM charges."
        )

    mm_charges = np.empty(len(mm_atoms), dtype=np.float64)
    for row, atom_index in enumerate(mm_atoms):
        charge, _, _ = nonbonded.getParticleParameters(int(atom_index))
        mm_charges[row] = charge.value_in_unit(unit.elementary_charge)

    return {
        "ml_atoms": ml_atoms,
        "mm_atoms": mm_atoms,
        "mm_charges": mm_charges,
    }


def _removeMLMMElectrostatics(system: openmm.System, mmInfo) -> None:
    """Zero ML-MM Coulomb in NonbondedForce while preserving the current LJ term."""
    if mmInfo is None:
        return

    ml_atoms = [int(i) for i in mmInfo["ml_atoms"]]
    mm_atoms = [int(i) for i in mmInfo["mm_atoms"]]
    if not ml_atoms or not mm_atoms:
        return

    for force in system.getForces():
        if not isinstance(force, openmm.NonbondedForce):
            continue

        num_particles = force.getNumParticles()
        atom_sigma = [None] * num_particles
        atom_epsilon = [None] * num_particles
        for i in range(num_particles):
            _, sigma, epsilon = force.getParticleParameters(i)
            atom_sigma[i] = sigma
            atom_epsilon[i] = epsilon

        exceptions = {}
        for i in range(force.getNumExceptions()):
            p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
            exceptions[(int(p1), int(p2))] = (chargeProd, sigma, epsilon)

        zero_charge_prod = 0.0 * unit.elementary_charge * unit.elementary_charge
        for ml_atom in ml_atoms:
            for mm_atom in mm_atoms:
                p1, p2 = (ml_atom, mm_atom) if ml_atom < mm_atom else (mm_atom, ml_atom)
                if (p1, p2) in exceptions:
                    _, sigma, epsilon = exceptions[(p1, p2)]
                else:
                    sigma = 0.5 * (atom_sigma[p1] + atom_sigma[p2])
                    epsilon = unit.sqrt(atom_epsilon[p1] * atom_epsilon[p2])
                force.addException(p1, p2, zero_charge_prod, sigma, epsilon, replace=True)



def _computeMACE(state, model, ptr, node_attrs, batch, pbc, returnEnergyType, charge, multiplicity, indices, periodic, linkInfo=None, mmInfo=None):
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
        from openmmml.embedding._links import compute_cap_positions, minimum_image_M
        r_Q = positions_full[linkInfo["q_global"]]
        r_M = positions_full[linkInfo["m_global"]]
        if periodic:
            # Minimum-image the Q->M bond so caps are placed correctly even if Q
            # and M sit across a periodic boundary. The same imaged r_M is reused
            # for force redistribution below, so the returned force stays the
            # gradient of the reported energy. Triclinic-correct; raises on a
            # genuinely wrapped pair (see minimum_image_M).
            cell_A = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
            r_M = minimum_image_M(r_Q, r_M, cell_A)
        pos_link, _C_L_unused = compute_cap_positions(
            r_Q, r_M, linkInfo["target_dist"]
        )
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
    if mmInfo is not None:
        mm_positions = positions_full[mmInfo["mm_atoms"]]
        inputDict["mm_positions"] = torch.tensor(
            mm_positions, dtype=dtype, device=ptr.device
        )
        inputDict["mm_charges"] = torch.tensor(
            mmInfo["mm_charges"], dtype=dtype, device=ptr.device
        )
        inputDict["mm_source_batch"] = torch.zeros(
            len(mmInfo["mm_atoms"]), dtype=torch.long, device=ptr.device
        )
    results = model(inputDict, compute_force=True)
    energy = float(results[returnEnergyType].detach())*energyScale
    forces = (results["forces"]*energyScale*lengthScale).detach().cpu().numpy()
    mm_forces = results.get("mm_forces")
    if mm_forces is not None:
        mm_forces = (mm_forces * energyScale * lengthScale).detach().cpu().numpy()
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=(np.float64 if dtype == torch.float64 else np.float32))
        if linkInfo is None:
            f[indices] = forces
        else:
            from openmmml.embedding._links import (
                compute_cap_positions,
                minimum_image_M,
                redistribute_cap_force,
            )
            N = len(indices)
            f_ml = forces[:N]
            f_link = forces[N:]
            f[indices] = f_ml
            # Redistribute each link atom's force onto its (Q, M) partners.
            # See _prepareLinkRecords for the bookkeeping. Use the SAME imaged
            # r_M as the cap placement above, or C_L / the bond direction would
            # not match the cap that produced f_link (breaking conservativeness
            # for boundary-crossing bonds).
            r_Q = positions_full[linkInfo["q_global"]]
            r_M = positions_full[linkInfo["m_global"]]
            if periodic:
                cell_A = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
                r_M = minimum_image_M(r_Q, r_M, cell_A)
            _, C_L = compute_cap_positions(r_Q, r_M, linkInfo["target_dist"])
            F_Q_add, F_M_add = redistribute_cap_force(f_link, r_Q, r_M, C_L)
            # q_global and m_global were validated unique at construction, so
            # plain += is correct (no duplicate-row aggregation needed).
            f[linkInfo["q_global"]] += F_Q_add.astype(f.dtype, copy=False)
            f[linkInfo["m_global"]] += F_M_add.astype(f.dtype, copy=False)
        if mmInfo is not None and mm_forces is not None:
            f[mmInfo["mm_atoms"]] += mm_forces.astype(f.dtype, copy=False)
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

    # Periodic systems are allowed: cap positions are placed with minimum-image
    # at runtime (_computeMACE), which is exact as long as each link bond is
    # shorter than half the box -- always true for real frontier bonds. A
    # genuinely wrapped (Q, M) pair (longer than half the box) raises there.
    if atoms is None:
        raise ValueError("linkRecords requires an explicit `atoms` subset.")

    if isinstance(linkRecords, (str, Path)):
        # capping_mapping.csv (from cli/oniom/cap_qm_boundary.py) is 1-based:
        # read (q_idx1, m_idx1, target_dist_ang) and convert to 0-based OpenMM
        # indices. target_dist stays in Angstroms (canonical — matches the
        # MACE-side `positions_full` in Å; the oniom-electrostatic closure
        # converts Å→nm at the point of use). Self-contained so openmm-ml does
        # not depend on mlmm for the CSV path. Cap positions are not stored in
        # the CSV; they are recomputed each step from (q, m, target_dist).
        import csv as _csv
        tuples = []
        with open(linkRecords, newline="") as _f:
            for row in _csv.DictReader(_f):
                tuples.append(
                    (int(row["q_idx1"]) - 1, int(row["m_idx1"]) - 1, float(row["target_dist_ang"]))
                )
    else:
        # Tuple path: target_dist is also in Å (matches the docstring
        # parameter name `target_dist_ang` and the MACE convention used
        # in every upstream test fixture, e.g. linkRecords=[(q, m, 1.09)]).
        tuples = [(int(q), int(m), float(td)) for (q, m, td) in linkRecords]

    if not tuples:
        return None

    num_particles = int(system.getNumParticles())
    atoms_set = set(int(a) for a in atoms)
    seen_pairs: set = set()
    seen_q: set = set()
    seen_m: set = set()
    q_global = np.empty(len(tuples), dtype=np.int64)
    m_global = np.empty(len(tuples), dtype=np.int64)
    target_dist = np.empty(len(tuples), dtype=np.float64)
    for k, (q, m, td) in enumerate(tuples):
        if not (0 <= q < num_particles):
            raise ValueError(f"linkRecords[{k}]: q_global={q} out of range [0, {num_particles}).")
        if not (0 <= m < num_particles):
            raise ValueError(f"linkRecords[{k}]: m_global={m} out of range [0, {num_particles}).")
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
