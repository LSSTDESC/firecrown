"""CMB source class and factory. Currently only supports CMB lensing convergence."""

from dataclasses import dataclass

import pyccl
import sacc
from pydantic import BaseModel, ConfigDict, PrivateAttr

from firecrown.likelihood._base import Source, Tracer
from firecrown.metadata_types import InferredGalaxyZDist, TypeSource
from firecrown.modeling_tools import ModelingTools


@dataclass(frozen=True)
class CMBConvergenceArgs:
    """Class for CMB convergence tracer arguments."""

    scale: float = 1.0
    field: str = "delta_matter"
    z_source: float = 1100.0


class CMBConvergence(Source):
    """Source class for CMB convergence lensing."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        scale: float = 1.0,
        z_source: float = 1100.0,
    ):
        """Initialize the CMBConvergence object.

        :param sacc_tracer: the name of the tracer in the SACC file.
        :param scale: the scale of the source.
        :param z_source: the source redshift for CMB lensing.
        """
        super().__init__(sacc_tracer)

        self.sacc_tracer = sacc_tracer
        self.scale = scale
        self.z_source = z_source
        self.current_tracer_args: None | CMBConvergenceArgs = None
        self.tracer_args: CMBConvergenceArgs

    def read_systematics(self, sacc_data: sacc.Sacc) -> None:
        """Read the systematics for this source from the SACC file.

        For CMB sources, there are no systematics to read.
        """

    def _read(self, sacc_data: sacc.Sacc) -> None:
        """Read the CMB tracer data from a sacc file."""
        self.tracer_args = CMBConvergenceArgs(
            scale=self.scale,
            field="delta_matter",
            z_source=self.z_source,
        )

        # For CMB, we just verify the tracer exists
        sacc_data.get_tracer(self.sacc_tracer)

    def get_scale(self) -> float:
        """Return the scale for this source."""
        current_args = self.current_tracer_args
        if current_args is None:
            raise RuntimeError("current_tracer_args is not initialized")
        return current_args.scale

    def create_tracers(self, tools: ModelingTools):
        """Create the CMB convergence tracer."""
        ccl_cosmo = tools.get_ccl_cosmology()
        tracer_args = self.tracer_args

        # Create CMB lensing tracer using z_source from tracer_args
        ccl_cmb_tracer = pyccl.CMBLensingTracer(
            ccl_cosmo, z_source=tracer_args.z_source
        )
        tracers = [
            Tracer(
                ccl_cmb_tracer, tracer_name="cmb_convergence", field=tracer_args.field
            )
        ]

        self.current_tracer_args = tracer_args
        return tracers, tracer_args

    @classmethod
    def create_ready(
        cls, sacc_tracer: str, scale: float = 1.0, z_source: float = 1100.0
    ) -> "CMBConvergence":
        """Create a CMBConvergence object ready for use."""
        obj = cls(sacc_tracer=sacc_tracer, scale=scale, z_source=z_source)
        obj.tracer_args = CMBConvergenceArgs(
            scale=scale, field="delta_matter", z_source=z_source
        )
        return obj


class CMBConvergenceFactory(BaseModel):
    """Factory class for CMBConvergence objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    _cache: dict[int, CMBConvergence] = PrivateAttr()

    type_source: TypeSource = TypeSource.DEFAULT
    z_source: float = 1100.0
    scale: float = 1.0

    def model_post_init(self, _, /) -> None:
        """Initialize the CMBConvergenceFactory."""
        self._cache: dict[int, CMBConvergence] = {}

    def create(self, inferred_galaxy_zdist: InferredGalaxyZDist) -> CMBConvergence:
        """Create a CMBConvergence object with the given inferred galaxy z distribution.

        :param inferred_galaxy_zdist: the inferred galaxy redshift distribution
        :return: a fully initialized CMBConvergence object
        """
        # Use the bin_name as the tracer identifier
        sacc_tracer = inferred_galaxy_zdist.bin_name
        tracer_id = hash(sacc_tracer)

        if tracer_id in self._cache:
            return self._cache[tracer_id]

        cmb_conv = CMBConvergence.create_ready(
            sacc_tracer=sacc_tracer, scale=self.scale, z_source=self.z_source
        )
        self._cache[tracer_id] = cmb_conv

        return cmb_conv

    def create_from_metadata_only(self, sacc_tracer: str) -> CMBConvergence:
        """Create a CMBConvergence object from metadata only.

        :param sacc_tracer: the name of the tracer
        :return: a fully initialized CMBConvergence object
        """
        tracer_id = hash(sacc_tracer)
        if tracer_id in self._cache:
            return self._cache[tracer_id]

        cmb_conv = CMBConvergence.create_ready(
            sacc_tracer=sacc_tracer, scale=self.scale, z_source=self.z_source
        )
        self._cache[tracer_id] = cmb_conv

        return cmb_conv
