from .parse import read_config, build_sources_from_config, parse_sacc
from .build import build_stats


def build_likelihood(build_parameters, sacc_data=None):
    config = build_parameters["config"]

    modeling_tools, cfg_w_classes = read_config(config)

    source_names, lens_names, tracer_combinations, sacc_data = parse_sacc(cfg_w_classes["data"], sacc_data)

    wl_sources, nc_sources = build_sources_from_config(source_names, lens_names, cfg_w_classes=cfg_w_classes)

    stats = build_stats(
        sources=wl_sources,
        lenses=nc_sources,
        tracer_combinations=tracer_combinations,
    )

    likelihood = cfg_w_classes["likelihood"](stats.values())

    likelihood.read(sacc_data)

    return likelihood, modeling_tools