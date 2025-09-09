from dataclasses import dataclass
import pyccl
import sacc
from firecrown.likelihood.source import SourceCMB, SourceCMBArgs, Tracer
from firecrown.modeling_tools import ModelingTools

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