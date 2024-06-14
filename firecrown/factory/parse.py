from typing import Callable, Optional
from dataclasses import dataclass

from jsonargparse import ArgumentParser

import sacc

from ..likelihood.gauss_family.statistic.source.weak_lensing import WeakLensing
from ..likelihood.gauss_family.statistic.source.number_counts import NumberCounts
from ..likelihood.gauss_family.statistic.source.source import SourceSystematic
from ..likelihood.gauss_family.statistic.statistic import Statistic
from ..modeling_tools import ModelingTools
from ..likelihood.likelihood import Likelihood

from .build import _build_sources, get_tracers


@dataclass
class SourceConfig:
    global_systematics: Optional[
        dict[str, SourceSystematic]
    ] = None

    per_bin_systematics: Optional[
        dict[
            str,
            Callable[
                [str], SourceSystematic
            ]
        ]
    ] = None


@dataclass
class DataConfig:
    statistics: Optional[
        dict[
            str, dict | None
        ]
    ]
    sacc: Optional[str] = None
    source_tracer_name: Optional[str] = "source"
    lens_tracer_name: Optional[str] = "lens"


def read_config(config_str):
    parser = ArgumentParser()
    parser.add_argument("two-point",
                        type=dict[str, SourceConfig])
    # parser.add_argument("cluster_counts",
    #                     type=ClusterConfig, fail_untyped=False)

    parser.add_class_arguments(ModelingTools, "modeling_tools")
    parser.add_argument("likelihood", type=Callable[[Statistic], Likelihood])

    parser.add_argument("data",
                        type=DataConfig)

    cfg = parser.parse_string(config_str)

    # Instatiate the classes, such as global systematics and modelling tools, which do
    # not depend on extra information
    cfg_w_classes = parser.instantiate_classes(cfg)

    return cfg_w_classes.get("modeling_tools", None), cfg_w_classes


def parse_sacc(data_config, sacc_data):
    if sacc_data is None:
        sacc_data = sacc.Sacc.load_fits(data_config.sacc)

    source_names, lens_names, tracer_combinations = get_tracers(
        sacc_data=sacc_data,
        statistics=data_config.statistics,
        source_id=data_config.source_tracer_name,
        lens_id=data_config.lens_tracer_name)

    return source_names, lens_names, tracer_combinations, sacc_data


def build_sources_from_config(source_names, lens_names, cfg_w_classes):
    # Get the classes in global_systematics directly, since they don't need additional information per-bin information
    source_global_systematics = {}
    if "weak_lensing" in cfg_w_classes["two-point"] and cfg_w_classes["two-point"]["weak_lensing"].global_systematics is not None:
        source_global_systematics = cfg_w_classes["two-point"]["weak_lensing"].global_systematics

    lens_global_systematics = {}
    if "number_counts" in cfg_w_classes["two-point"] and cfg_w_classes["two-point"]["number_counts"].global_systematics is not None:
        lens_global_systematics = cfg_w_classes["two-point"]["number_counts"].global_systematics

    # Instatiate the per_bin_systematics, providing the sacc_tracer argument
    source_per_bin_systematics = {}
    if "weak_lensing" in cfg_w_classes["two-point"] and cfg_w_classes["two-point"]["weak_lensing"].per_bin_systematics is not None:
        source_per_bin_systematics = cfg_w_classes["two-point"]["weak_lensing"].per_bin_systematics

    lens_per_bin_systematics = {}
    if "number_counts" in cfg_w_classes["two-point"] and cfg_w_classes["two-point"]["number_counts"].per_bin_systematics is not None:
        lens_per_bin_systematics = cfg_w_classes["two-point"]["number_counts"].per_bin_systematics

    # Build the WeakLensing and NumberCounts Source objects
    wl_sources = _build_sources(
        source_names,
        source_class=WeakLensing,
        per_bin_systematics=source_per_bin_systematics.values(),
        global_systematics=source_global_systematics.values()
    )

    nc_sources = _build_sources(
        lens_names,
        source_class=NumberCounts,
        per_bin_systematics=lens_per_bin_systematics.values(),
        global_systematics=lens_global_systematics.values()
    )

    return wl_sources, nc_sources
