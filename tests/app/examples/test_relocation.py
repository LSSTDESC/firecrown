"""Tests that generated examples do not record where they or Firecrown are.

A generated example must run after its directory is moved, or copied to a machine
where Firecrown is installed elsewhere, when the framework is run from the example
directory.
"""

import configparser
import importlib.util
from pathlib import Path
from typing import Any

import pytest

import firecrown
from firecrown.app.analysis import AnalysisBuilder, Frameworks
from firecrown.app.analysis._cosmosis import CONNECTOR_SHIM, CONNECTOR_SHIM_SOURCE
from firecrown.app.examples import ExampleCMBCross, ExampleCosmicShear
from firecrown.connector.cosmosis import likelihood as cosmosis_likelihood

EXAMPLES: dict[str, tuple[type[AnalysisBuilder], dict[str, Any]]] = {
    "cosmic_shear": (ExampleCosmicShear, {}),
    "cmb_cross": (ExampleCMBCross, {"n_bins": 2, "n_ell_points": 5, "n_z_points": 100}),
}
CASES = [(name, framework) for name in EXAMPLES for framework in Frameworks]


@pytest.fixture(
    name="generated",
    params=CASES,
    ids=[f"{name}-{framework.value}" for name, framework in CASES],
)
def fixture_generated(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[Path, Frameworks, str]:
    """Generate each example for each framework with the default options."""
    name, framework = request.param
    example_cls, options = EXAMPLES[name]
    output = tmp_path / "example"
    example_cls(output_path=output, prefix=name, target_framework=framework, **options)
    return output, framework, name


def test_no_absolute_locations(generated: tuple[Path, Frameworks, str]) -> None:
    """No generated text file contains the output directory or the Firecrown path."""
    output, _, _ = generated
    locations = {
        output.absolute().as_posix(),
        output.resolve().as_posix(),
        Path(firecrown.__path__[0]).resolve().as_posix(),
    }
    text_files = [
        path
        for path in output.iterdir()
        if path.suffix in {".ini", ".yaml", ".yml", ".py"}
    ]
    assert text_files
    for path in text_files:
        text = path.read_text(encoding="utf-8")
        for location in locations:
            assert location not in text, f"{path.name} contains {location}"


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_cosmosis_loads_connector_shim(name: str, tmp_path: Path) -> None:
    """The CosmoSIS pipeline loads the shim next to the ini file, without root."""
    example_cls, options = EXAMPLES[name]
    output = tmp_path / "example"
    example_cls(
        output_path=output,
        prefix=name,
        target_framework=Frameworks.COSMOSIS,
        **options,
    )

    cfg = configparser.ConfigParser(allow_no_value=True, interpolation=None)
    cfg.read(output / f"cosmosis_{name}.ini")

    assert "root" not in cfg["runtime"]
    assert cfg["firecrown_likelihood"]["file"] == CONNECTOR_SHIM
    assert (output / CONNECTOR_SHIM).read_text(encoding="utf-8") == (
        CONNECTOR_SHIM_SOURCE
    )


def test_connector_shim_exports_the_connector(tmp_path: Path) -> None:
    """Importing the shim gives the setup, execute and cleanup of the connector."""
    shim = tmp_path / CONNECTOR_SHIM
    shim.write_text(CONNECTOR_SHIM_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("firecrown_cosmosis", shim)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.setup is cosmosis_likelihood.setup
    assert module.execute is cosmosis_likelihood.execute
    assert module.cleanup is cosmosis_likelihood.cleanup


def test_absolute_paths_point_to_the_shim(tmp_path: Path) -> None:
    """With absolute paths, the ini names the shim in the output directory."""
    output = tmp_path / "example"
    ExampleCosmicShear(
        output_path=output,
        prefix="cosmic_shear",
        target_framework=Frameworks.COSMOSIS,
        use_absolute_path=True,
    )
    cfg = configparser.ConfigParser(allow_no_value=True, interpolation=None)
    cfg.read(output / "cosmosis_cosmic_shear.ini")

    assert cfg["firecrown_likelihood"]["file"] == (
        (output / CONNECTOR_SHIM).absolute().as_posix()
    )
