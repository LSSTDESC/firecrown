"""Tests for the CosmoSIS likelihood module."""

from cosmosis.datablock import DataBlock
from pytest import fixture

import firecrown.connector.cosmosis.likelihood as like


@fixture(name="config")
def make_config() -> DataBlock:
    """Return a DataBlock with a minimal configuration."""
    db = DataBlock()
    db.put("module_options", "firecrown_config", "tests/likelihood/lkdir/lkscript.py")
    return db


def test_module_cleanup(config: DataBlock):
    """The module's cleanup function just returns zero."""
    m = like.setup(config)
    status = like.cleanup(m)
    assert status == 0


def test_module_init(config: DataBlock):
    module = like.setup(config)
    assert isinstance(module, like.FirecrownLikelihood)
