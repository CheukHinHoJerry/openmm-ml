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
from openmmml.embeddings import utilities
from typing import Iterable, Optional
from functools import partial
import numpy as np


def _floating_reference(module):
    for tensor in module.buffers():
        if tensor.is_floating_point():
            return tensor
    for tensor in module.parameters():
        if tensor.is_floating_point():
            return tensor
    return None


def _rebuild_feature_block(block, pbc_handling: str = "auto"):
    """Rebuild a deterministic graph block saved by an older graph release."""
    from graph_longrange.features import GTOElectrostaticFeatures

    realspace = block.realspace_features
    quadrupoles = bool(
        getattr(
            getattr(block.non_periodic_correction_terms, "self_field", None),
            "include_quadrupole_corrections",
            False,
        )
    )
    rebuilt = GTOElectrostaticFeatures(
        density_max_l=int(realspace.density_max_l),
        density_smearing_width=float(realspace.density_smearing_width),
        feature_max_l=int(realspace.projection_max_l),
        feature_smearing_widths=[
            float(x) for x in realspace.projection_smearing_widths
        ],
        include_self_interaction=bool(block.include_self_interaction),
        kspace_cutoff=float(block.kspace_cutoff),
        quadrupole_feature_corrections=quadrupoles,
        integral_normalization=str(block.feature_basis.normalize),
        pbc_handling=pbc_handling,
    )
    reference = _floating_reference(block)
    if reference is not None:
        rebuilt = rebuilt.to(device=reference.device, dtype=reference.dtype)
    return rebuilt


def _rebuild_energy_block(block, pbc_handling: str = "auto"):
    from graph_longrange.energy import GTOElectrostaticEnergy

    rebuilt = GTOElectrostaticEnergy(
        density_max_l=int(block.density_max_l),
        density_smearing_width=float(block.density_smearing_width),
        kspace_cutoff=float(block.kspace_cutoff),
        include_self_interaction=bool(block.include_self_interaction),
        pbc_handling=pbc_handling,
    )
    reference = _floating_reference(block)
    if reference is not None:
        rebuilt = rebuilt.to(device=reference.device, dtype=reference.dtype)
    return rebuilt


def _prepare_external_sources(model, data, compute_force: bool):
    import torch

    positions = data.get("mm_positions")
    charges = data.get("mm_charges")
    if (
        positions is None
        or charges is None
        or positions.numel() == 0
        or charges.numel() == 0
    ):
        return None

    ml_positions = data["positions"]
    positions = positions.to(device=ml_positions.device, dtype=ml_positions.dtype)
    positions = positions.clone().requires_grad_(compute_force)
    width = (int(model.atomic_multipoles_max_l) + 1) ** 2
    features = torch.zeros(
        (charges.numel(), width),
        dtype=ml_positions.dtype,
        device=ml_positions.device,
    )
    features[:, 0] = charges.to(features).reshape(-1)
    if positions.shape[0] != features.shape[0]:
        raise ValueError(
            "MM positions and electrostatic sources must have the same length."
        )

    transform = getattr(model, "_charges_to_mul_ir", None)
    if transform is not None:
        features = transform(features)

    batch = data.get("mm_source_batch")
    if batch is None:
        if int(data["pbc"].reshape(-1, 3).shape[0]) != 1:
            raise ValueError(
                "mm_source_batch is required for batched PolarMACE inputs."
            )
        batch = torch.zeros(
            positions.shape[0], dtype=torch.long, device=positions.device
        )
    else:
        batch = batch.to(device=positions.device, dtype=torch.long).reshape(-1)
    if batch.shape[0] != positions.shape[0]:
        raise ValueError(
            "mm_source_batch and mm_positions must have the same length."
        )
    return {"positions": positions, "features": features, "batch": batch}


def _enable_polarmace_external_sources(model):
    """Wrap PolarMACE with dynamic MM electrostatic sources in eager mode."""
    import torch

    if getattr(model, "supports_external_electrostatics", False):
        return model
    if model.__class__.__name__ != "PolarMACE":
        raise TypeError(
            "External electrostatic sources require a PolarMACE model; got "
            f"{model.__class__.__name__}."
        )

    try:
        from graph_longrange.external_source_energy import (
            GTOElectrostaticExternalSourceEnergy,
        )
        from graph_longrange.external_source_features import (
            GTOElectrostaticExternalSourceFeatures,
        )
    except ImportError as exc:
        raise ImportError(
            "PolarMACE electrostatic embedding requires a graph_longrange "
            "release that provides the external-source energy and feature blocks."
        ) from exc

    feature_base = _rebuild_feature_block(model.electric_potential_descriptor)
    energy_base = _rebuild_energy_block(model.coulomb_energy)
    model.electric_potential_descriptor = (
        GTOElectrostaticExternalSourceFeatures.from_features(
            feature_base,
            # PolarMACE has two spin channels. Each receives half of the
            # physical external potential.
            external_scale=0.5,
        )
    )
    model.coulomb_energy = GTOElectrostaticExternalSourceEnergy.from_energy(
        energy_base
    )

    class PolarMACEExternalSources(torch.nn.Module):
        supports_external_electrostatics = True

        def __init__(self, wrapped):
            super().__init__()
            self.model = wrapped

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.model, name)

        def forward(
            self,
            data,
            training: bool = False,
            compute_force: bool = True,
            compute_virials: bool = False,
            compute_stress: bool = False,
            compute_displacement: bool = False,
            compute_hessian: bool = False,
            compute_edge_forces: bool = False,
            compute_atomic_stresses: bool = False,
            **kwargs,
        ):
            external = _prepare_external_sources(self.model, data, compute_force)
            if external is None:
                return self.model(
                    data,
                    training=training,
                    compute_force=compute_force,
                    compute_virials=compute_virials,
                    compute_stress=compute_stress,
                    compute_displacement=compute_displacement,
                    compute_hessian=compute_hessian,
                    compute_edge_forces=compute_edge_forces,
                    compute_atomic_stresses=compute_atomic_stresses,
                    **kwargs,
                )
            if any(
                (
                    compute_virials,
                    compute_stress,
                    compute_displacement,
                    compute_hessian,
                    compute_edge_forces,
                    compute_atomic_stresses,
                )
            ):
                raise NotImplementedError(
                    "The OpenMM external-source adapter currently supports "
                    "energies and Cartesian forces only."
                )

            external_kwargs = {
                "external_feats": external["features"],
                "external_positions": external["positions"],
                "external_batch": external["batch"],
            }
            self.model.electric_potential_descriptor.set_external_sources(
                **external_kwargs
            )
            self.model.coulomb_energy.set_external_sources(**external_kwargs)
            try:
                result = self.model(
                    data,
                    training=training,
                    compute_force=False,
                    compute_virials=False,
                    compute_stress=False,
                    compute_displacement=False,
                    compute_hessian=False,
                    compute_edge_forces=False,
                    compute_atomic_stresses=False,
                    **kwargs,
                )
                if compute_force:
                    ml_gradient, mm_gradient = torch.autograd.grad(
                        outputs=[result["energy"]],
                        inputs=[data["positions"], external["positions"]],
                        grad_outputs=[torch.ones_like(result["energy"])],
                        create_graph=training,
                        retain_graph=training,
                        allow_unused=True,
                    )
                    result["forces"] = (
                        torch.zeros_like(data["positions"])
                        if ml_gradient is None
                        else -ml_gradient
                    )
                    result["mm_forces"] = (
                        torch.zeros_like(external["positions"])
                        if mm_gradient is None
                        else -mm_gradient
                    )
                else:
                    result["mm_forces"] = None
                return result
            finally:
                self.model.electric_potential_descriptor.clear_external_sources()
                self.model.coulomb_energy.clear_external_sources()

    return PolarMACEExternalSources(model)


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
            model = _enable_polarmace_external_sources(model)
        return model, device

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        precision: Optional[str] = None,
        returnEnergyType: str = "energy",
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


def _computeMACE(state, model, ptr, node_attrs, batch, pbc, returnEnergyType, charge, multiplicity, indices, periodic, mmInfo=None):
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

    # Scatter ML and MM forces back to the full system.
    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=(np.float64 if dtype == torch.float64 else np.float32))
        f[indices] = forces
        if mmInfo is not None and mm_forces is not None:
            f[mmInfo["mm_atoms"]] += mm_forces.astype(f.dtype, copy=False)
        forces = f
    return energy, forces
