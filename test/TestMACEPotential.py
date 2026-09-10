import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
import torch

from openmmml import MLPotential
from openmmml.models.macepotential import (
    MACEPotentialImpl,
    _computeMACE,
    _prepareMMEmbedding,
    _should_use_mm_embedding,
)

mace = pytest.importorskip("mace", reason="mace is not installed")
platform_ints = range(mm.Platform.getNumPlatforms())
# Get the path to the test data
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _simple_nonbonded_system():
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    params = [
        (1.0, 0.30, 0.20),
        (-0.5, 0.40, 0.50),
        (0.25, 0.50, 0.80),
    ]
    for charge_e, sigma_nm, epsilon_kj in params:
        system.addParticle(12.0)
        nonbonded.addParticle(
            charge_e * unit.elementary_charge,
            sigma_nm * unit.nanometer,
            epsilon_kj * unit.kilojoule_per_mole,
        )
    system.addForce(nonbonded)
    return system, nonbonded


def testPrepareMMEmbedding():
    """The MM complement and its charges are read off the NonbondedForce."""
    system, nonbonded = _simple_nonbonded_system()
    info = _prepareMMEmbedding(system, [0, 1])
    assert info is not None
    np.testing.assert_array_equal(info["ml_atoms"], [0, 1])
    np.testing.assert_array_equal(info["mm_atoms"], [2])
    np.testing.assert_allclose(info["mm_charges"], [0.25], atol=1e-12)


class _FakeState:
    def __init__(self, positions_angstrom):
        self._positions = np.asarray(positions_angstrom, dtype=np.float64) * unit.angstrom

    def getPositions(self, asNumpy=False):
        return self._positions


class _FakeModel:
    def __init__(self, dtype=torch.float32):
        self.r_max = torch.tensor(3.0, dtype=dtype)
        self.dtype = dtype

    def __call__(self, input_dict, compute_force=True):
        del compute_force
        n_ml = input_dict["positions"].shape[0]
        forces = torch.tensor(
            [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]][:n_ml],
            dtype=self.dtype,
            device=input_dict["positions"].device,
        )
        out = {
            "interaction_energy": torch.tensor([2.5], dtype=self.dtype, device=forces.device),
            "forces": forces,
        }
        if "mm_positions" in input_dict:
            out["mm_forces"] = torch.tensor(
                [[0.5, 0.25, -0.75]],
                dtype=self.dtype,
                device=forces.device,
            )
        return out


class PolarMACE:
    supports_external_electrostatics = True


class MACE:
    pass


def testComputeMACEScattersMMForces():
    ptr = torch.tensor([0, 2], dtype=torch.long)
    node_attrs = torch.ones((2, 1), dtype=torch.float32)
    batch = torch.zeros(2, dtype=torch.long)
    pbc = torch.tensor([False, False, False], dtype=torch.bool)
    mm_info = {"mm_atoms": np.array([2], dtype=np.int64), "mm_charges": np.array([0.25], dtype=np.float64)}

    energy, forces = _computeMACE(
        state=_FakeState([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        model=_FakeModel(dtype=torch.float32),
        ptr=ptr,
        node_attrs=node_attrs,
        batch=batch,
        pbc=pbc,
        returnEnergyType="interaction_energy",
        charge=torch.tensor([0.0], dtype=torch.float32),
        multiplicity=torch.tensor([1.0], dtype=torch.float32),
        indices=np.array([0, 1], dtype=np.int64),
        periodic=False,
        linkInfo=None,
        mmInfo=mm_info,
    )

    assert energy == pytest.approx(2.5 * 96.4853)
    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
            [0.5, 0.25, -0.75],
        ],
        dtype=np.float32,
    ) * (96.4853 * 10.0)
    np.testing.assert_allclose(forces, expected, rtol=1e-6, atol=1e-6)


def testShouldUseMMEmbedding():
    assert _should_use_mm_embedding(PolarMACE(), [0, 1], "electrostatic")
    assert not _should_use_mm_embedding(PolarMACE(), [0, 1], "mechanical")
    assert not _should_use_mm_embedding(MACE(), [0, 1], "mechanical")
    with pytest.raises(ValueError, match="Unsupported embedding mode"):
        _should_use_mm_embedding(PolarMACE(), [0, 1], "bad-mode")


def testElectrostaticEmbeddingRejectsUnsupportedCases():
    """Electrostatic embedding must fail loudly rather than falling back to
    mechanical embedding: by the time the potential is asked to add its forces,
    the ML-MM electrostatics have already been removed from the MM force field,
    so a silent fallback would simply lose them."""
    with pytest.raises(ValueError, match="requires a model that accepts MM charges"):
        _should_use_mm_embedding(MACE(), [0, 1], "electrostatic")
    with pytest.raises(ValueError, match="requires an ML subset"):
        _should_use_mm_embedding(PolarMACE(), None, "electrostatic")


def testOnlyPolarModelsOfferElectrostaticEmbedding():
    """Electrostatic embedding needs a model that accepts MM charges, which of
    the pretrained models only the PolarMACE family does.  A custom checkpoint
    may be one, so it is offered there too and checked once the model loads."""
    for name, (_, _, _, _, acceptsMMCharges) in MACEPotentialImpl.KNOWN_MODELS.items():
        embeddings = MLPotential(name).getSupportedEmbeddings()
        assert ('electrostatic' in embeddings) == acceptsMMCharges, name
        assert acceptsMMCharges == name.startswith('mace-polar'), name
    custom = MLPotential('mace', modelPath='unused-until-forces-are-added.model')
    assert 'electrostatic' in custom.getSupportedEmbeddings()


@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMACE:

    # Reference energies are calculated with MACECalculator
    refEnergy = {
        'mace-off23-small': -713468.6327560507,
        'mace-off23-medium': -713468.0563706581,
        'mace-off23-large': -713467.7476380612,
        'mace-off24-medium': -713467.9394350434,
        'mace-mpa-0-medium': -8839.299589829867,
        'mace-omat-0-small': -8726.63865431241,
        'mace-omat-0-medium': -8679.026847088873,
        'mace-omol-0-extra-large': -712903.4934289698,
        'mace-les-off-small': -713467.9354591698,
        'mace-polar-1-small': -712903.1710073923,
        'mace-polar-1-medium': -712903.4536792638,
        'mace-polar-1-large': -712903.7834631138,
    }

    @pytest.mark.parametrize("model", ['mace-off23-small', 'mace-off23-medium', 'mace-off23-large', 'mace-off24-medium',
                                       'mace-mpa-0-medium', 'mace-omat-0-small', 'mace-omat-0-medium', 'mace-omol-0-extra-large',
                                       'mace-les-off-small', 'mace-polar-1-small', 'mace-polar-1-medium', 'mace-polar-1-large'])
    def testCreatePureMLSystem(self, platform_int, model):
        if 'mace-les' in model:
            pytest.importorskip("les", reason="les is not installed")
        if 'mace-polar' in model:
            pytest.importorskip("graph_longrange", reason="graph_electrostatics is not installed")
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(self.refEnergy[model], energyML, rtol=1e-6)

    def testPeriodicSystem(self, platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential("mace-off23-small")
        system = potential.createSystem(pdb.topology, returnEnergyType='energy')
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -151723354.26015 # Calculated with MACECalculator
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            energyML = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            assert np.isclose(energyRef, energyML, rtol=1e-5)

    def testCreateMixedSystem(self, platform_int):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("mace-off23-small")
        mixedSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=False)
        interpSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=True)
        platform = mm.Platform.getPlatform(platform_int)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(inpcrd.positions)
        mixedContext.setPositions(inpcrd.positions)
        interpContext.setPositions(inpcrd.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpContext.setParameter('lambda_interpolate', 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=1e-5)
        assert np.isclose(mmEnergy, interpEnergy2, rtol=1e-5)

    @pytest.mark.parametrize("precision", ["single", "double"])
    def testPrecisionApplied(self, platform_int, precision):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential('mace-off23-small')

        # Specifying precision single/double.
        system = potential.createSystem(pdb.topology, returnEnergyType='energy', precision=precision)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.positions)

        # Inconsistent dtypes will crash the simulation.
        energyML = context.getState(energy=True, forces=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # The energy should be physically meaningful under both precisions
        assert np.isfinite(energyML), \
            "Energy is not finite under precision {}".format(precision)
        assert np.isclose(energyML, self.refEnergy['mace-off23-small'], rtol=1e-6),\
            "Energy is not close to reference under precision {}".format(precision)


def testMissingMMForcesIsAnError():
    """A model given MM charges must return forces on the MM atoms."""
    class _NoMMForces(_FakeModel):
        def __call__(self, inputDict, compute_force=True):
            out = super().__call__(inputDict, compute_force)
            out.pop("mm_forces", None)
            return out

    mmInfo = {
        "mm_atoms": np.array([2], dtype=np.int64),
        "mm_charges": np.array([0.25], dtype=np.float64),
    }
    with pytest.raises(ValueError, match="no 'mm_forces'"):
        _computeMACE(
            state=_FakeState(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
            ),
            model=_NoMMForces(dtype=torch.float32),
            ptr=torch.tensor([0, 2], dtype=torch.long),
            node_attrs=torch.ones((2, 1), dtype=torch.float32),
            batch=torch.zeros(2, dtype=torch.long),
            pbc=torch.tensor([False, False, False], dtype=torch.bool),
            returnEnergyType="interaction_energy",
            charge=torch.tensor([0.0], dtype=torch.float32),
            multiplicity=torch.tensor([1.0], dtype=torch.float32),
            indices=np.array([0, 1], dtype=np.int64),
            periodic=False,
            linkInfo=None,
            mmInfo=mmInfo,
        )
