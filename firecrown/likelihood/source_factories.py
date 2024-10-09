"""Factory functions for creating sources."""

from firecrown.likelihood.number_counts import NumberCountsFactory, NumberCounts
from firecrown.likelihood.weak_lensing import WeakLensingFactory, WeakLensing
from firecrown.metadata_types import InferredGalaxyZDist, Measurement, Galaxies


def use_source_factory(
    inferred_galaxy_zdist: InferredGalaxyZDist,
    measurement: Measurement,
    wl_factory: WeakLensingFactory | None = None,
    nc_factory: NumberCountsFactory | None = None,
) -> WeakLensing | NumberCounts:
    """Apply the factory to the inferred galaxy redshift distribution."""
    source: WeakLensing | NumberCounts
    if measurement not in inferred_galaxy_zdist.measurements:
        raise ValueError(
            f"Measurement {measurement} not found in inferred galaxy redshift "
            f"distribution {inferred_galaxy_zdist.bin_name}!"
        )

    match measurement:
        case Galaxies.COUNTS:
            assert nc_factory is not None
            source = nc_factory.create(inferred_galaxy_zdist)
        case (
            Galaxies.SHEAR_E
            | Galaxies.SHEAR_T
            | Galaxies.SHEAR_MINUS
            | Galaxies.SHEAR_PLUS
        ):
            assert wl_factory is not None
            source = wl_factory.create(inferred_galaxy_zdist)
        case _:
            raise ValueError(f"Measurement {measurement} not supported!")
    return source


def use_source_factory_metadata_index(
    sacc_tracer: str,
    measurement: Measurement,
    wl_factory: WeakLensingFactory | None = None,
    nc_factory: NumberCountsFactory | None = None,
) -> WeakLensing | NumberCounts:
    """Apply the factory to create a source from metadata only."""
    source: WeakLensing | NumberCounts
    match measurement:
        case Galaxies.COUNTS:
            assert nc_factory is not None
            source = nc_factory.create_from_metadata_only(sacc_tracer)
        case (
            Galaxies.SHEAR_E
            | Galaxies.SHEAR_T
            | Galaxies.SHEAR_MINUS
            | Galaxies.SHEAR_PLUS
        ):
            assert wl_factory is not None
            source = wl_factory.create_from_metadata_only(sacc_tracer)
        case _:
            raise ValueError(f"Measurement {measurement} not supported!")
    return source
