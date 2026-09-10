"""Small eager-mode adapter for PolarMACE external electrostatic sources.

The electrostatic mathematics lives in graph_longrange.  This module only
bridges dynamic MM sources into an otherwise unchanged PolarMACE forward pass.
It is intended for OpenMM's PythonForce execution path, not TorchScript export.
"""

from __future__ import annotations


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
    multipoles = data.get("mm_multipoles")
    if charges is not None and multipoles is not None:
        raise ValueError("mm_charges and mm_multipoles are mutually exclusive.")
    values = multipoles if multipoles is not None else charges
    if positions is None or values is None or positions.numel() == 0 or values.numel() == 0:
        return None

    ml_positions = data["positions"]
    positions = positions.to(device=ml_positions.device, dtype=ml_positions.dtype)
    positions = positions.clone().requires_grad_(compute_force)
    width = (int(model.atomic_multipoles_max_l) + 1) ** 2
    if multipoles is None:
        features = torch.zeros(
            (charges.numel(), width), dtype=ml_positions.dtype, device=ml_positions.device
        )
        features[:, 0] = charges.to(features).reshape(-1)
    else:
        features = multipoles.to(device=ml_positions.device, dtype=ml_positions.dtype).clone()
        if features.dim() != 2 or features.shape != (positions.shape[0], width):
            raise ValueError(f"mm_multipoles must have shape [N_mm, {width}].")
        if width >= 4:
            # Public Cartesian (q, px, py, pz) -> graph/e3nn (q, py, pz, px).
            features[:, 1:4] = features[:, [2, 3, 1]]
    if positions.shape[0] != features.shape[0]:
        raise ValueError("MM positions and electrostatic sources must have the same length.")

    transform = getattr(model, "_charges_to_mul_ir", None)
    if transform is not None:
        features = transform(features)

    batch = data.get("mm_source_batch")
    if batch is None:
        if int(data["pbc"].reshape(-1, 3).shape[0]) != 1:
            raise ValueError("mm_source_batch is required for batched PolarMACE inputs.")
        batch = torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device)
    else:
        batch = batch.to(device=positions.device, dtype=torch.long).reshape(-1)
    if batch.shape[0] != positions.shape[0]:
        raise ValueError("mm_source_batch and mm_positions must have the same length.")
    return {"positions": positions, "features": features, "batch": batch}


def enable_polarmace_external_sources(model):
    """Return an eager wrapper that adds dynamic MM electrostatic sources.

    Models that have already been wrapped are returned unchanged.  Non-PolarMACE
    models are rejected rather than silently dropping the external interaction.
    """
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
    feature_block = GTOElectrostaticExternalSourceFeatures.from_features(
        feature_base,
        # PolarMACE has two spin channels. Each receives half of the physical
        # external potential.
        external_scale=0.5,
    )
    energy_block = GTOElectrostaticExternalSourceEnergy.from_energy(energy_base)
    model.electric_potential_descriptor = feature_block
    model.coulomb_energy = energy_block

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
                (compute_virials, compute_stress, compute_displacement, compute_hessian,
                 compute_edge_forces, compute_atomic_stresses)
            ):
                raise NotImplementedError(
                    "The OpenMM external-source adapter currently supports energies and "
                    "Cartesian forces only."
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
                        if ml_gradient is None else -ml_gradient
                    )
                    result["mm_forces"] = (
                        torch.zeros_like(external["positions"])
                        if mm_gradient is None else -mm_gradient
                    )
                else:
                    result["mm_forces"] = None
                return result
            finally:
                self.model.electric_potential_descriptor.clear_external_sources()
                self.model.coulomb_energy.clear_external_sources()

    return PolarMACEExternalSources(model)


__all__ = ["enable_polarmace_external_sources"]
