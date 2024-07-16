import sacc

from ..likelihood.gauss_family.statistic.two_point import TwoPoint


def get_tracers(sacc_data, statistics=None, source_id="source", lens_id="lens"):
    source_names = set()
    lens_names = set()

    if statistics is None:
        statistics = sacc_data.get_data_types()

    tracer_combinations = {}

    for data_type in statistics:
        tracer_combinations[data_type] = []
        tracers = sacc_data.get_tracer_combinations(data_type)
        for tracer_names in tracers:
            tracer_types = []
            for tracer_name in tracer_names:
                if source_id in tracer_name:
                    source_names.add(tracer_name)
                    tracer_types.append("shear")
                elif lens_id in tracer_name:
                    lens_names.add(tracer_name)
                    tracer_types.append("density")
                else:
                    raise ValueError(f"Unknown tracer type {tracer_name}!")

            tracer_combinations[data_type].append((tracer_names, tracer_types))

    return source_names, lens_names, tracer_combinations


def _build_sources(
    source_names, source_class, per_bin_systematics=None, global_systematics=None
):
    if per_bin_systematics is None:
        per_bin_systematics = []

    if global_systematics is None:
        global_systematics = []

    global_source_systematics = []

    for systematic in global_systematics:
        global_source_systematics.append(systematic)

    sources = {}

    for tracer_name in source_names:
        per_bin_source_systematics = []
        for systematic in per_bin_systematics:
            per_bin_source_systematics.append(systematic(tracer_name))

        sources[tracer_name] = source_class(
            sacc_tracer=tracer_name,
            systematics=global_source_systematics + per_bin_source_systematics,
        )

    return sources


COSMIC_SHEAR_TYPES = ["galaxy_shear_cl_ee", "galaxy_shear_xi_plus", "galaxy_shear_xi_minus"]
GALAXY_CLUSTERING_TYPES = ["galaxy_density_cl", "galaxy_density_xi"]
GGL_TYPES = ["galaxy_shearDensity_cl_e", "galaxy_shearDensity_xi_t"]


def build_stats(sources, lenses, tracer_combinations, statistics=None, scale_cut_func=None):
    stats = {}

    sources = sources if sources is not None else {}
    lenses = lenses if lenses is not None else {}

    if statistics is None:
        statistics = tracer_combinations.keys()

    if scale_cut_func is None:

        def scale_cut_func(_):
            return (None, None)

    for statistic in statistics:
        if statistic not in tracer_combinations:
            raise ValueError(f"Statistic {statistic} not available. Valid options are {list(tracer_combinations.keys())}")
        for (tracer_1_name, tracer_2_name), (tracer_1_type, tracer_2_type) in tracer_combinations[statistic]:
            if statistic in COSMIC_SHEAR_TYPES:
                # Cosmic shear
                source_1 = sources[tracer_1_name]
                source_2 = sources[tracer_2_name]
            elif statistic in GALAXY_CLUSTERING_TYPES:
                # Galaxy clustering
                source_1 = lenses[tracer_1_name]
                source_2 = lenses[tracer_2_name]
            elif GGL_TYPES:
                # GGL
                if tracer_1_type == "shear":
                    source_1 = sources[tracer_1_name]
                    source_2 = lenses[tracer_2_name]
                else:
                    source_1 = lenses[tracer_1_name]
                    source_2 = sources[tracer_2_name]
            else:
                raise ValueError(f"{statistic=} not supported")

            ell_or_theta_min_max = scale_cut_func((tracer_1_name, tracer_2_name))
            if ell_or_theta_min_max is None:
                print(
                    f"No overlap between redshift kernels for "
                    f"{tracer_1_name}-{tracer_2_name}"
                )
                continue
            else:
                ell_or_theta_min, ell_or_theta_max = ell_or_theta_min_max

            stats[f"{statistic}_{tracer_1_name}_{tracer_2_name}"] = TwoPoint(
                source0=source_1,
                source1=source_2,
                sacc_data_type=statistic,
                ell_or_theta_min=ell_or_theta_min,
                ell_or_theta_max=ell_or_theta_max,
            )

    return stats

