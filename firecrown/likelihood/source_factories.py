"""Factory functions for creating sources."""

from firecrown.likelihood.number_counts import NumberCountsFactory, NumberCounts
from firecrown.likelihood.weak_lensing import WeakLensingFactory, WeakLensing
from firecrown.metadata_types import InferredGalaxyZDist, Measurement, Galaxies


def use_source_factory(
    inferred_galaxy_zdist: InferredGalaxyZDist,
    measurement: Measurement,
    factories: list[WeakLensingFactory | NumberCountsFactory],
) -> WeakLensing | NumberCounts:
    """Apply the appropriate factory to the inferred galaxy redshift distribution."""
    source: WeakLensing | NumberCounts
    if measurement not in inferred_galaxy_zdist.measurements:
        raise ValueError(
            f"Measurement {measurement} not found in inferred galaxy redshift "
            f"distribution {inferred_galaxy_zdist.bin_name}!"
        )

    for factory in factories:
        match measurement:
            case Galaxies.COUNTS:
                if isinstance(factory, NumberCountsFactory):
                    source = factory.create(inferred_galaxy_zdist)
                    break
            case (
                Galaxies.SHEAR_E
                | Galaxies.SHEAR_T
                | Galaxies.SHEAR_MINUS
                | Galaxies.SHEAR_PLUS
            ):
                if isinstance(factory, WeakLensingFactory):
                    source = factory.create(inferred_galaxy_zdist)
                    break
            case _:
                raise ValueError(f"Measurement {measurement} not supported!")
    else:
        raise ValueError(f"No suitable factory found for measurement {measurement}!")


def use_source_factory_metadata_index(
    sacc_tracer: str,
    measurement: Measurement,
    factories: list[WeakLensingFactory | NumberCountsFactory],
) -> WeakLensing | NumberCounts:
    """Apply the appropriate factory to create a source from metadata only."""
    source: WeakLensing | NumberCounts
    for factory in factories:
        match measurement:
            case Galaxies.COUNTS:
                if isinstance(factory, NumberCountsFactory):
                    source = factory.create_from_metadata_only(sacc_tracer)
                    break
            case (
                Galaxies.SHEAR_E
                | Galaxies.SHEAR_T
                | Galaxies.SHEAR_MINUS
                | Galaxies.SHEAR_PLUS
            ):
                if isinstance(factory, WeakLensingFactory):
                    source = factory.create_from_metadata_only(sacc_tracer)
                    break
            case _:
                raise ValueError(f"Measurement {measurement} not supported!")
    else:
        raise ValueError(f"No suitable factory found for measurement {measurement}!")
    return source
