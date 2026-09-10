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
import os
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from openmmml.embeddings import utilities
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
    'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium', 'mace-omol-0-extra-large',
    'mace-les-off-small', and the PolarMACE models 'mace-polar-1-small',
    'mace-polar-1-medium', and 'mace-polar-1-large'.  The PolarMACE models are
    the ones that support electrostatic embedding.

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
    which are not in ``interaction_energy``.

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

    # (Function name, model name, restrictive license name or None, long-range,
    # accepts MM charges)
    #
    # The last flag records whether the model can be given the charges and
    # positions of the atoms outside the ML subset, which is what electrostatic
    # embedding requires.  Only the PolarMACE family can.
    KNOWN_MODELS = {
        'mace-off23-small': ('mace_off', 'small', 'ASL', False, False),
        'mace-off23-medium': ('mace_off', 'medium', 'ASL', False, False),
        'mace-off23-large': ('mace_off', 'large', 'ASL', False, False),
        'mace-off24-medium': ('mace_off', 'https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true', 'ASL', False, False),
        'mace-mpa-0-medium': ('mace_mp', 'medium-mpa-0', None, False, False),
        'mace-omat-0-small': ('mace_mp', 'small-omat-0', 'ASL', False, False),
        'mace-omat-0-medium': ('mace_mp', 'medium-omat-0', 'ASL', False, False),
        'mace-omol-0-extra-large': ('mace_omol', 'extra_large', 'ASL', False, False),
        'mace-les-off-small': ('mace_off', 'https://github.com/ChengUCB/les_fit/blob/main/MACELES-OFF/MACELES-OFF_small_converted.model?raw=true', 'CC BY-NC 4.0', True, False),
        'mace-polar-1-small': ('mace_polar', 'polar-1-s', None, True, True),
        'mace-polar-1-medium': ('mace_polar', 'polar-1-m', None, True, True),
        'mace-polar-1-large': ('mace_polar', 'polar-1-l', None, True, True),
    }

    def __init__(self, name: str, modelPath) -> None:
        """
        Initialize the MACEPotentialImpl.

        Parameters
        ----------
        name : str
            The name of the MACE model.
            Options include 'mace-off23-small', 'mace-off23-medium', 'mace-off23-large',
            'mace-off24-medium', 'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium',
            'mace-omol-0-extra-large', 'mace-les-off-small', 'mace-polar-1-small',
            'mace-polar-1-medium', 'mace-polar-1-large', and 'mace'.
        modelPath : str, optional
            The path to the locally trained MACE model if ``name`` is 'mace'.
        """
        self.name = name
        self.modelPath = modelPath
        self._preloadedModel = None

    def _loadModel(self, args):
        """Load the MACE model, returning it along with the device it is on.

        If createMixedSystem() has already loaded a model in order to inspect
        it, that one is handed over here rather than the checkpoint being read a
        second time.  The handover is consumed on use, so each call after that
        loads afresh; holding the model indefinitely would mean a later call
        with a different precision converting an already converted model.
        """
        import torch
        try:
            from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol, mace_polar
        except ImportError as e:
            raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")

        device = self._getTorchDevice(args)
        preloaded, self._preloadedModel = self._preloadedModel, None
        if preloaded is not None and preloaded[1] == device:
            return preloaded[0], device

        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            functions = {
                'mace_off': mace_off,
                'mace_mp': mace_mp,
                'mace_omol': mace_omol,
                'mace_polar': mace_polar,
            }
            fnName, name, restrictiveLicense, _, _ = MACEPotentialImpl.KNOWN_MODELS[self.name]
            model = functions[fnName](model=name, device=device, return_raw_model=True).to(device)
            if restrictiveLicense is not None:
                import logging
                logging.warning(f'The model {self.name} is distributed under the restrictive {restrictiveLicense} license.  Commercial use is not permitted.')
        elif self.name == "mace":
            if self.modelPath is not None:
                model = torch.load(self.modelPath, map_location=device)
                if hasattr(model, "to"):
                    model = model.to(device)
            else:
                raise ValueError("No modelPath provided for local MACE model.")
        else:
            raise ValueError(f"Unsupported MACE model: {self.name}")
        if model.__class__.__name__ == "PolarMACE":
            from openmmml.models._polarmace_external import (
                enable_polarmace_external_sources,
            )

            model = enable_polarmace_external_sources(model)
        return model, device

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
        customNonbondedChargeParameter: Optional[str] = None,
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
            Which embedding method the caller is implementing. Set by
            ``createMixedSystem``; there is normally no reason to pass it here
            directly. ``mechanical`` (the default) does not pass MM positions
            or charges into MACE. ``electrostatic`` passes them into PolarMACE
            and scatters the returned ``mm_forces`` back onto the MM atoms; the
            removal of the classical ML-MM Coulomb that this assumes is done by
            ``createMixedSystem``, not here. It is an error to request it for a
            model that cannot accept MM charges.
        """
        import torch
        try:
            from mace.tools import utils, to_one_hot, atomic_numbers_to_indices
            from mace.calculators.foundations_models import mace_off, mace_mp, mace_omol, mace_polar
        except ImportError as e:
            raise ImportError(f"Failed to import mace with error: {e}. Install mace with 'pip install mace-torch'.")
        try:
            from e3nn.util import jit
        except ImportError as e:
            raise ImportError(f"Failed to import e3nn with error: {e}. Install e3nn with 'pip install e3nn'.")

        assert returnEnergyType in ["interaction_energy", "energy"], f"Unsupported returnEnergyType: '{returnEnergyType}'. Supported options are 'interaction_energy' or 'energy'."

        # Load the model.

        model, device = self._loadModel(args)

        use_mm_embedding = _should_use_mm_embedding(model, atoms, embedding)

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomicNumbers = [atom.element.atomic_number for atom in includedAtoms]

        if returnEnergyType == "energy":
            try:
                e0_max = float(model.atomic_energies_fn.atomic_energies.detach().abs().max())
            except AttributeError:
                e0_max = 0.0
            if e0_max > 100.0:  # 1 eV per atom is conservative; foundation models far exceed this
                import warnings as _w
                _w.warn(
                    f"returnEnergyType='energy' includes per-atom reference "
                    f"energies (max |e0| = {e0_max:.2f} eV/atom over {int(model.atomic_energies_fn.atomic_energies.numel())} "
                    f"element entries). Use precision='double' if you need accurate absolute "
                    f"energies, or pass returnEnergyType='interaction_energy' for the "
                    f"e0-subtracted readout (note: only 'energy' is gradient-consistent "
                    f"with PolarMACE — see the docstring).",
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
            mmInfo = _prepareMMEmbedding(system, atoms, customNonbondedChargeParameter)

            # Optional Z1 / DZ1 link-atom charge redistribution. Standard QM/MM
            # correction to stop the QM region from being over-polarised by
            # the partial charge on the MM-side boundary atom.
            if linkInfo is not None and linkChargeScheme not in (None, "none"):
                from openmmml.models._links import (
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
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

    def getMLLongRange(self) -> bool | None:
        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            _, _, _, longRange, _ = MACEPotentialImpl.KNOWN_MODELS[self.name]
            return longRange
        return None

    def getSupportedEmbeddings(self) -> list[str]:

        # Electrostatic embedding requires a model that accepts the charges and
        # positions of the atoms outside the ML subset, which of the pretrained
        # models only the PolarMACE family does.  A custom checkpoint may be a
        # PolarMACE model too, but that cannot be known without loading it, so
        # the method is offered and createMixedSystem() rejects the checkpoint
        # once loaded if it turns out not to be one.

        if self.name in MACEPotentialImpl.KNOWN_MODELS:
            _, _, _, _, acceptsMMCharges = MACEPotentialImpl.KNOWN_MODELS[self.name]
            return ["electrostatic"] if acceptsMMCharges else []
        return ["electrostatic"]

    def createMixedSystem(self,
                          topology: openmm.app.Topology,
                          system: openmm.System,
                          atoms: list[int],
                          forceGroup: int,
                          interpolate: bool,
                          embedding: str,
                          customNonbondedHasCharges: Optional[bool] = None,
                          customNonbondedChargeParameter: Optional[str] = None,
                          **args) -> openmm.System:
        """Create a mixed system using electrostatic embedding.

        The model, rather than the conventional force field, is responsible for
        the electrostatic interactions between the atoms within the ML subset
        and those outside of it: it is passed the positions and conventional
        force field charges of the atoms outside the ML subset, and returns
        forces on them alongside the forces on the ML subset.  The ML subset can
        therefore polarize in response to its surroundings, which mechanical
        embedding does not allow.

        This is implemented as the "global charge zero" variant: the
        conventional force field charge of every atom in the ML subset is set to
        zero.  Every Coulomb term involving an ML atom is then zero by
        construction, including the reciprocal space part of PME, without any
        per-pair exceptions being added.  Adding an exception for each ML-MM
        pair would instead be incorrect under periodic boundary conditions,
        since NonbondedForce evaluates exceptions using plain Cartesian
        distances rather than the minimum image convention, so the ML-MM
        Lennard-Jones interaction would silently vanish for any pair that is
        only within the cutoff across a periodic boundary.  Lennard-Jones is
        left to the conventional force field and continues to use the ordinary,
        periodicity-aware pair list.

        Interactions within the ML subset are excluded entirely, as the model
        computes them.  Bonded terms that cross the ML/MM boundary are retained.

        Only models that accept MM charges and positions, which at present means
        the PolarMACE family, can be used with this embedding method.  An error
        is raised for any other model rather than falling back to mechanical
        embedding, since by that point the ML-MM electrostatics have already
        been removed from the conventional force field and a fallback would
        simply lose them.

        Because this method has to account for every Coulomb term in the force
        field, it requires the System to contain exactly one NonbondedForce: the
        MM charges given to the model are read from one, so several would be
        ambiguous.

        It also needs to be told about any CustomNonbondedForce, whose energy
        expression is arbitrary and cannot be inspected here.  Pass
        customNonbondedHasCharges=False to declare that it holds no
        electrostatics, or True together with customNonbondedChargeParameter
        naming the per-particle parameter that holds the charge, which is then
        zeroed on the ML atoms exactly as for the NonbondedForce.  An error is
        raised if the answer is needed and has not been given.

        Note that zeroing that parameter removes the ML terms only if the
        expression is multiplicatively separable in the charge, as the usual
        q1*q2/r is.  That cannot be verified here, so it is the caller's
        responsibility.
        """

        if embedding != "electrostatic":
            raise ValueError(f"Unsupported embedding type: {embedding}")

        # Check that the model can actually accept MM charges and positions
        # before touching the System, so that an unsuitable model is rejected
        # with the System left alone rather than stripped of its ML-MM
        # electrostatics.  This is also the first point at which the check is
        # possible, since it needs the loaded checkpoint; the model is handed to
        # addForces() below so the checkpoint is only read once.

        model, device = self._loadModel(args)
        if not _supports_mm_embedding(model):
            raise ValueError(
                f"embedding='{embedding}' requires a model that accepts MM charges "
                f"and positions (PolarMACE); got {model.__class__.__name__}."
            )

        if interpolate:
            # At lambda_interpolate=0 the conventional endpoint would be missing
            # the ML-MM Coulomb energy, which is removed from the conventional
            # force field outside of the interpolating CustomCVForce and cannot
            # be restored from within it.
            raise ValueError("Electrostatic embedding does not support interpolation.")

        periodic = system.usesPeriodicBoundaryConditions()

        # Electrostatic embedding has to account for every Coulomb term in the
        # force field: the ones involving the ML subset are removed here on the
        # understanding that the model supplies them.  Anything it cannot see is
        # either left in place and counted twice, or removed and never replaced,
        # and in both cases the result is a wrong energy rather than an error.
        # So refuse the cases where the electrostatics cannot be located rather
        # than guessing.

        nonbondedForces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if len(nonbondedForces) > 1:
            # The MM charges handed to the model are read from a single
            # NonbondedForce, so several of them are ambiguous.
            raise ValueError("Multiple NonbondedForce objects encountered; electrostatic embedding requires exactly one.")

        if any(isinstance(f, openmm.CustomNonbondedForce) for f in system.getForces()):
            # A CustomNonbondedForce's energy expression is arbitrary, so
            # whether it contains electrostatics cannot be determined here.
            if customNonbondedHasCharges is None:
                raise ValueError("The System contains a CustomNonbondedForce and it is unknown whether it includes electrostatic interactions; pass customNonbondedHasCharges to specify.")
            if customNonbondedHasCharges and customNonbondedChargeParameter is None:
                raise ValueError("A CustomNonbondedForce includes electrostatic interactions, so customNonbondedChargeParameter must name the per-particle parameter holding the charge.")

        # Create the new system with the ML-ML interactions that the model
        # computes removed.

        newSystem = utilities.removeBonds(system, topology, atoms, True)
        atomSet = set(atoms)

        for force in newSystem.getForces():
            if isinstance(force, openmm.NonbondedForce):

                # Zero the charge of every ML atom, which removes all Coulomb
                # interactions involving the ML subset while leaving its
                # Lennard-Jones parameters, and the MM-MM interactions,
                # untouched.

                for atom in atoms:
                    charge, sigma, epsilon = force.getParticleParameters(atom)
                    force.setParticleParameters(atom, 0.0, sigma, epsilon)

                # setParticleParameters() does not update the charge products
                # that were precomputed for existing exceptions, so the 1-4
                # Coulomb terms crossing the ML/MM boundary have to be zeroed
                # separately.

                for index in range(force.getNumExceptions()):
                    p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(index)
                    if p1 in atomSet or p2 in atomSet:
                        force.setExceptionParameters(index, p1, p2, 0.0, sigma, epsilon)

                # Exclude the ML-ML interactions entirely.

                for i in range(len(atoms)):
                    for j in range(i):
                        force.addException(atoms[i], atoms[j], 0, 1, 0, True)

                force.setExceptionsUsePeriodicBoundaryConditions(periodic)

            elif isinstance(force, openmm.CustomNonbondedForce):

                # Zero the named charge parameter on the ML atoms, the same
                # trick used for the NonbondedForce above.  Unlike there it is
                # not guaranteed to work: it removes the ML terms only if the
                # energy expression is multiplicatively separable in the charge,
                # as q1*q2/r is, and that cannot be checked here.

                if customNonbondedChargeParameter is not None:
                    names = [force.getPerParticleParameterName(i)
                             for i in range(force.getNumPerParticleParameters())]
                    if customNonbondedChargeParameter not in names:
                        raise ValueError(f"A CustomNonbondedForce has no per-particle parameter '{customNonbondedChargeParameter}'; it defines {names}.")
                    chargeIndex = names.index(customNonbondedChargeParameter)
                    for atom in atoms:
                        parameters = list(force.getParticleParameters(atom))
                        parameters[chargeIndex] = 0.0
                        force.setParticleParameters(atom, parameters)

                utilities.addCustomNonbondedExclusions(force, atoms)

        # Add the ML potential, telling it that it is responsible for the
        # electrostatic interactions with the atoms outside the ML subset, and
        # handing over the model already loaded above.

        self._preloadedModel = (model, device)
        try:
            self.addForces(topology, newSystem, atoms, forceGroup, embedding=embedding,
                       customNonbondedChargeParameter=customNonbondedChargeParameter, **args)
        finally:
            self._preloadedModel = None

        return newSystem


def _supports_mm_embedding(model) -> bool:
    return bool(getattr(model, "supports_external_electrostatics", False))


_SUPPORTED_EMBEDDINGS = ("mechanical", "electrostatic")
_MM_EMBEDDING_MODES = ("electrostatic",)


def _should_use_mm_embedding(model, atoms: Optional[Iterable[int]], embedding: str) -> bool:
    if embedding not in _SUPPORTED_EMBEDDINGS:
        raise ValueError(
            f"Unsupported embedding mode '{embedding}'. Supported values are "
            + ", ".join(repr(m) for m in _SUPPORTED_EMBEDDINGS)
            + "."
        )
    if embedding not in _MM_EMBEDDING_MODES:
        return False
    if not _supports_mm_embedding(model):
        # The mixed system has had its ML-MM Coulomb removed on the assumption
        # that the model will supply it, so falling back to mechanical
        # embedding here would silently discard those interactions.
        raise ValueError(
            f"embedding='{embedding}' requires a model that accepts MM charges "
            f"and positions (PolarMACE); got {model.__class__.__name__}."
        )
    if atoms is None:
        raise ValueError(
            f"embedding='{embedding}' requires an ML subset; it cannot be used "
            "with createSystem()."
        )
    return True


def _prepareMMEmbedding(system: openmm.System, atoms: Optional[Iterable[int]],
                        customNonbondedChargeParameter: Optional[str] = None):
    """Extract the MM complement and its charges from the system's NonbondedForce."""
    if atoms is None:
        return None

    num_particles = int(system.getNumParticles())
    ml_atoms = np.asarray(list(atoms), dtype=np.int64)
    ml_set = set(int(i) for i in ml_atoms.tolist())
    mm_atoms = np.asarray(
        [i for i in range(num_particles) if i not in ml_set], dtype=np.int64
    )

    # The charges given to the model have to come from wherever the force field
    # actually keeps them, which is the same force createMixedSystem zeroed the
    # ML charges in.  When that is a CustomNonbondedForce the caller has named
    # the parameter holding them.
    mm_charges = np.empty(len(mm_atoms), dtype=np.float64)

    if customNonbondedChargeParameter is not None:
        custom = None
        for force in system.getForces():
            if isinstance(force, openmm.CustomNonbondedForce):
                names = [force.getPerParticleParameterName(i)
                         for i in range(force.getNumPerParticleParameters())]
                if customNonbondedChargeParameter in names:
                    custom = force
                    chargeIndex = names.index(customNonbondedChargeParameter)
                    break
        if custom is None:
            raise ValueError(f"No CustomNonbondedForce defines a per-particle parameter '{customNonbondedChargeParameter}'.")
        for row, atom_index in enumerate(mm_atoms):
            mm_charges[row] = custom.getParticleParameters(int(atom_index))[chargeIndex]
        return {
            "ml_atoms": ml_atoms,
            "mm_atoms": mm_atoms,
            "mm_charges": mm_charges,
        }

    nonbonded = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = force
            break
    if nonbonded is None:
        raise ValueError(
            "PolarMACE MM embedding requires a NonbondedForce to source MM charges."
        )

    for row, atom_index in enumerate(mm_atoms):
        charge, _, _ = nonbonded.getParticleParameters(int(atom_index))
        mm_charges[row] = charge.value_in_unit(unit.elementary_charge)

    return {
        "ml_atoms": ml_atoms,
        "mm_atoms": mm_atoms,
        "mm_charges": mm_charges,
    }


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
        from openmmml.models._links import compute_cap_positions, minimum_image_M
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
    # load mm position and charges
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
    if mmInfo is not None and mm_forces is None:
        # The mixed system has had its ML-MM electrostatics removed on the
        # understanding that this model supplies them.  A model that returns no
        # forces on the MM atoms did not compute them, so continuing would leave
        # those interactions missing entirely rather than merely approximated,
        # and nothing downstream would report it.  The usual cause is a
        # PolarMACE checkpoint whose forward does not accept mm_charges.
        raise ValueError("The model returned no 'mm_forces' although MM charges were supplied; it does not implement electrostatic embedding.")
    if mm_forces is not None:
        mm_forces = (mm_forces * energyScale * lengthScale).detach().cpu().numpy()

    # force redistribution and add mm_forces back to the full system
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=(np.float64 if dtype == torch.float64 else np.float32))
        if linkInfo is None:
            f[indices] = forces
        else:
            from openmmml.models._links import (
                compute_cap_positions,
                minimum_image_M,
                redistribute_cap_force,
            )
            N = len(indices)
            f_ml = forces[:N]
            f_link = forces[N:]
            f[indices] = f_ml

            # Redistribute each link atom's force onto its (Q, M) partners.
            r_Q = positions_full[linkInfo["q_global"]]
            r_M = positions_full[linkInfo["m_global"]]
            if periodic:
                cell_A = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
                r_M = minimum_image_M(r_Q, r_M, cell_A)
            _, C_L = compute_cap_positions(r_Q, r_M, linkInfo["target_dist"])
            F_Q_add, F_M_add = redistribute_cap_force(f_link, r_Q, r_M, C_L)

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
        # A capping-mapping CSV is 1-based: read (q_idx1, m_idx1,
        # target_dist_ang) and convert to 0-based OpenMM indices. target_dist
        # stays in Angstroms, canonical because it matches the MACE-side
        # `positions_full`. Cap positions are not stored in the CSV; they are
        # recomputed each step from (q, m, target_dist).
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
