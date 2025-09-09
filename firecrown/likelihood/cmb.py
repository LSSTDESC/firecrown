from dataclasses import dataclass
from typing import Sequence
import pyccl
import sacc
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from firecrown.likelihood.source import SourceCMB, SourceCMBArgs, Tracer
from firecrown.modeling_tools import ModelingTools
from firecrown.metadata_types import TypeSource

@dataclass(frozen=True)
class CMBConvergenceArgs(SourceCMBArgs):
    """Class for CMB convergence tracer arguments."""
    z_source: float = 1100.0  # Add z_source as a field

class CMBConvergence(SourceCMB):
    """Source class for CMB convergence lensing."""

    def __init__(
        self,
        *,
        sacc_tracer: str,
        scale: float = 1.0,
        z_source: float = 1100.0,  # Add z_source parameter
    ):
        """Initialize the CMBConvergence object."""
        super().__init__(sacc_tracer=sacc_tracer, scale=scale)
        self.z_source = z_source  # Store z_source
        
    def _read(self, sacc_data: sacc.Sacc) -> None:
        """Read the data for this source from the SACC file."""
        self.tracer_args = CMBConvergenceArgs(
            scale=self.scale,
            field="delta_matter",
            z_source=self.z_source  # Include z_source in tracer_args
        )
        super()._read(sacc_data)

    def create_tracers(self, tools: ModelingTools):
        """Create the CMB convergence tracer."""
        ccl_cosmo = tools.get_ccl_cosmology()
        tracer_args = self.tracer_args

        # Create CMB lensing tracer using z_source from tracer_args
        ccl_cmb_tracer = pyccl.CMBLensingTracer(ccl_cosmo, z_source=tracer_args.z_source)
        tracers = [Tracer(ccl_cmb_tracer, tracer_name="cmb_convergence", field=tracer_args.field)]

        self.current_tracer_args = tracer_args
        return tracers, None

    @classmethod
    def create_ready(cls, sacc_tracer: str, scale: float = 1.0, z_source: float = 1100.0) -> "CMBConvergence":
        """Create a CMBConvergence object ready for use."""
        obj = cls(sacc_tracer=sacc_tracer, scale=scale, z_source=z_source)
        obj.tracer_args = CMBConvergenceArgs(scale=scale, field="delta_matter", z_source=z_source)
        return obj

class CMBConvergenceFactory(BaseModel):
    """Factory class for CMBConvergence objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    _cache: dict[int, CMBConvergence] = PrivateAttr()
    
    type_source: TypeSource = TypeSource.DEFAULT
    z_source: float = 1100.0

    # adding these fields to match the structure of other factories
    global_systematics: Sequence[object] = []   # CMB doesn't have global systematics, but keep for consistency or future use

    def model_post_init(self, _, /) -> None:
        """Initialize the CMBConvergenceFactory."""
        self._cache: dict[int, CMBConvergence] = {}

    def create(self, sacc_tracer: str) -> CMBConvergence:
        """Create a CMBConvergence object with the given tracer name.

        :param sacc_tracer: the name of the tracer
        :return: a fully initialized CMBConvergence object
        """
        tracer_id = hash(sacc_tracer)
        if tracer_id in self._cache:
            return self._cache[tracer_id]

        cmb_conv = CMBConvergence.create_ready(
            sacc_tracer=sacc_tracer, 
            scale=self.scale, 
            z_source=self.z_source
        )
        self._cache[tracer_id] = cmb_conv

        return cmb_conv

    def create_from_metadata_only(self, sacc_tracer: str) -> CMBConvergence:
        """Create a CMBConvergence object from metadata only.

        :param sacc_tracer: the name of the tracer
        :return: a fully initialized CMBConvergence object
        """
        return self.create(sacc_tracer)