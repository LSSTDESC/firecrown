from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import CCLFactory
from firecrown.updatable import get_default_params_map
from firecrown.parameters import ParamsMap
from firecrown.likelihood.factories import (
    build_two_point_likelihood,
    DataSourceSacc,
    ensure_path,
    TwoPointCorrelationSpace,
    TwoPointExperiment,
    TwoPointFactory,
)
weak_lensing_yaml = """
per_bin_systematics:
- type: MultiplicativeShearBiasFactory
- type: PhotoZShiftFactory
global_systematics:
- type: LinearAlignmentSystematicFactory
  alphag: 1.0
"""
number_counts_yaml = """
per_bin_systematics:
- type: PhotoZShiftFactory
global_systematics: []
"""
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.weak_lensing import WeakLensingFactory
from firecrown.likelihood.number_counts import NumberCountsFactory
from firecrown.utils import base_model_from_yaml
from firecrown.data_functions import TwoPointBinFilterCollection, TwoPointBinFilter
from firecrown.modeling_tools import ModelingTools
from firecrown.metadata_types import Galaxies
weak_lensing_factory = base_model_from_yaml(WeakLensingFactory, weak_lensing_yaml)
number_counts_factory = base_model_from_yaml(NumberCountsFactory, number_counts_yaml)

tpf = TwoPointFactory(
                correlation_space=TwoPointCorrelationSpace.REAL,
                weak_lensing_factory=weak_lensing_factory,
                number_counts_factory=number_counts_factory,
            )

two_point_experiment = TwoPointExperiment(
    two_point_factory=tpf,
    data_source=DataSourceSacc(
        sacc_data_file="../examples/des_y1_3x2pt/sacc_data.fits",
        filters=TwoPointBinFilterCollection(
            filters=[
                TwoPointBinFilter.from_args_auto(
                    name=f"lens{i}",
                    measurement=Galaxies.COUNTS,
                    lower=0.5,
                    upper=300,
                )
                for i in range(5)
            ],
            require_filter_for_all=False,
            allow_empty=True,
        ),
    ),
)


two_point_experiment_filtered = TwoPointExperiment(
    two_point_factory=tpf,
    data_source=DataSourceSacc(
        sacc_data_file="../examples/des_y1_3x2pt/sacc_data.fits",
        filters=TwoPointBinFilterCollection(
            filters=[
                TwoPointBinFilter.from_args_auto(
                    name=f"lens{i}",
                    measurement=Galaxies.COUNTS,
                    lower=2999,
                    upper=3000,
                )
                for i in range(5)
            ],
            require_filter_for_all=False,
            allow_empty=True,
        ),
    ),
)

tools = ModelingTools(ccl_factory=CCLFactory(require_nonlinear_pk=True))

likelihood_tpe = two_point_experiment.make_likelihood()

params = get_default_params_map(tools, likelihood_tpe)

tools = ModelingTools()
tools.update(params)
tools.prepare()
likelihood_tpe.update(params)

likelihood_tpe_filtered = two_point_experiment_filtered.make_likelihood()

params = get_default_params_map(tools, likelihood_tpe_filtered)

tools = ModelingTools()
tools.update(params)
tools.prepare()
likelihood_tpe_filtered.update(params)


print(f"Loglike from TwoPointExperiment: {likelihood_tpe.compute_loglike(tools)}")
print(f"Loglike from filtered TwoPointExperiment: {likelihood_tpe_filtered.compute_loglike(tools)}")