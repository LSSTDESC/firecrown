"""Unit tests for firecrown.app.examples._sn_cosmic_shear module.

Tests ExampleSNCosmicShear example generator and build_likelihood execution.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from firecrown.app.analysis import FrameworkCosmology, Frameworks
from firecrown.app.examples import EXAMPLES_LIST, _sn_cosmic_shear_template
from firecrown.app.examples._sn_cosmic_shear import ExampleSNCosmicShear
from firecrown.app.examples._sn_srd import ExampleSupernovaSRD
from firecrown.app.sacc import SaccFormat, Transform
from firecrown.likelihood import (
    ConstGaussian,
    NamedParameters,
    SumLikelihood,
    load_likelihood_from_module_type,
)
from firecrown.modeling_tools import ModelingTools


def _builder(tmp_path: Path, **kwargs: object) -> ExampleSNCosmicShear:
    """Create the builder without running the generation."""
    with patch.object(ExampleSNCosmicShear, "_proceed_generation"):
        return ExampleSNCosmicShear(
            output_path=tmp_path,
            target_framework=Frameworks.COSMOSIS,
            **kwargs,  # type: ignore[arg-type]
        )


def test_registered() -> None:
    """The example is available as sn_cosmic_shear."""
    assert EXAMPLES_LIST["sn_cosmic_shear"] is ExampleSNCosmicShear


def test_default_prefix(tmp_path: Path) -> None:
    """The default prefix is sn_cosmic_shear."""
    assert _builder(tmp_path).prefix == "sn_cosmic_shear"


@pytest.mark.parametrize("sacc_format", SaccFormat)
def test_sn_sacc_path(tmp_path: Path, sacc_format: SaccFormat) -> None:
    """The supernova file name ends with the target format."""
    builder = _builder(tmp_path, prefix="t", sacc_format=sacc_format)

    assert builder.sn_sacc_path(tmp_path) == tmp_path / f"t_sn.{sacc_format.value}"


@pytest.mark.parametrize("sacc_format", SaccFormat)
def test_generate_sacc_converts_sn_file(
    tmp_path: Path, sacc_format: SaccFormat
) -> None:
    """The supernova file is written in the target format, in both branches."""
    builder = _builder(tmp_path, prefix="t", sacc_format=sacc_format, n_ell_points=3)

    shear_sacc = builder.generate_sacc(tmp_path)

    sn_sacc = builder.sn_sacc_path(tmp_path)
    assert shear_sacc == tmp_path / "t.sacc"
    assert Transform.detect_format(sn_sacc) == sacc_format
    assert not (tmp_path / "t_sn.sacc").exists()


def test_generate_sacc_downloads_sn_data(tmp_path: Path) -> None:
    """The supernova data come from the SRD URL."""
    builder = _builder(tmp_path, prefix="t")

    with (
        patch(
            "firecrown.app.examples._sn_cosmic_shear.download_from_url"
        ) as mock_download,
        patch("firecrown.app.examples._sn_cosmic_shear.Transform") as mock_transform,
        patch(
            "firecrown.app.examples._cosmic_shear.ExampleCosmicShear.generate_sacc",
            return_value=tmp_path / "t.sacc",
        ),
    ):
        mock_transform.detect_format.return_value = builder.sacc_format
        mock_download.side_effect = lambda _url, out, *_: out.touch()
        builder.generate_sacc(tmp_path)

    assert mock_download.call_args[0][0] == ExampleSupernovaSRD.data_url


def test_generate_factory_copies_template(tmp_path: Path) -> None:
    """The factory file is named after the prefix."""
    builder = _builder(tmp_path, prefix="t")

    result = builder.generate_factory(tmp_path, tmp_path / "t.sacc")

    assert result == tmp_path / "t_factory.py"
    assert result.is_file()


@pytest.mark.parametrize("use_absolute_path", [True, False])
def test_get_build_parameters(tmp_path: Path, use_absolute_path: bool) -> None:
    """Both SACC paths and n_bins are passed to the factory."""
    builder = _builder(
        tmp_path, prefix="t", n_bins=3, use_absolute_path=use_absolute_path
    )
    shear_sacc = tmp_path / "t.hdf5"

    params = builder.get_build_parameters(shear_sacc).convert_to_basic_dict()

    sn_sacc = builder.sn_sacc_path(tmp_path)
    assert params == {
        "sn_sacc_file": builder.get_sacc_file(sn_sacc),
        "shear_sacc_file": builder.get_sacc_file(shear_sacc),
        "n_bins": 3,
    }


def test_get_models(tmp_path: Path) -> None:
    """One model holds the delta_z parameters and the supernova magnitude."""
    builder = _builder(tmp_path, prefix="t", n_bins=2)

    (model,) = builder.get_models()

    assert model.name == "firecrown_t"
    assert [p.name for p in model.parameters] == [
        "trc0_delta_z",
        "trc1_delta_z",
        "sn_ddf_sample_M",
    ]


def test_required_cosmology(tmp_path: Path) -> None:
    """Cosmic shear requires the nonlinear power spectrum."""
    assert _builder(tmp_path).required_cosmology() == FrameworkCosmology.NONLINEAR


def test_get_options_desc(tmp_path: Path) -> None:
    """The options list the supernova data and the cosmic shear options."""
    options = dict(_builder(tmp_path, n_bins=3).get_options_desc())

    assert options["Supernova data"] == ExampleSupernovaSRD.data_url
    assert options["Number of Tomographic Bins"] == "3"


def test_build_likelihood_execution(tmp_path: Path) -> None:
    """The template and its copy return two likelihoods that the loader sums."""
    builder = _builder(tmp_path, prefix="t", n_bins=2, n_ell_points=5)
    shear_sacc = builder.generate_sacc(tmp_path)
    factory_file = builder.generate_factory(tmp_path, shear_sacc)
    params = NamedParameters(
        {
            "sn_sacc_file": str(builder.sn_sacc_path(tmp_path)),
            "shear_sacc_file": str(shear_sacc),
            "n_bins": 2,
        }
    )

    likelihoods, tools = _sn_cosmic_shear_template.build_likelihood(params)
    assert isinstance(tools, ModelingTools)
    assert len(likelihoods) == 2
    sn_likelihood, shear_likelihood = likelihoods
    assert isinstance(sn_likelihood, ConstGaussian)
    assert isinstance(shear_likelihood, ConstGaussian)
    assert len(shear_likelihood.statistics) == 3  # n_bins * (n_bins + 1) / 2

    spec = importlib.util.spec_from_file_location("t_factory", factory_file)
    assert spec is not None
    assert spec.loader is not None
    factory_module = importlib.util.module_from_spec(spec)
    sys.modules["t_factory"] = factory_module
    try:
        spec.loader.exec_module(factory_module)
        likelihood, _ = load_likelihood_from_module_type(factory_module, params)
    finally:
        del sys.modules["t_factory"]

    assert isinstance(likelihood, SumLikelihood)
    assert len(likelihood.likelihoods) == 2


def test_full_generation(tmp_path: Path) -> None:
    """The full workflow writes both SACC files, the factory and the configuration."""
    ExampleSNCosmicShear(
        output_path=tmp_path,
        prefix="t",
        n_ell_points=3,
        target_framework=Frameworks.COSMOSIS,
    )

    assert (tmp_path / "t.hdf5").is_file()
    assert (tmp_path / "t_sn.hdf5").is_file()
    assert (tmp_path / "t_factory.py").is_file()
    assert (tmp_path / "cosmosis_t.ini").is_file()
