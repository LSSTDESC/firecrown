"""
Tests for function supporting SACC.

"""
from firecrown.sacc_support import BinZTracer, BinRichnessTracer, BinRadiusTracer


def test_make_binztracer():
    tracer = BinZTracer.make("bin_z", name="fred", z_lower=0.5, z_upper=1.0)
    assert isinstance(tracer, BinZTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "fred"
    assert tracer.z_lower == 0.5
    assert tracer.z_upper == 1.0


def test_binztracer_tables():
    a = BinZTracer.make("bin_z", name="fred", z_lower=0.5, z_upper=1.0)
    b = BinZTracer.make("bin_z", name="wilma", z_lower=1.0, z_upper=1.5)
    tables = BinZTracer.to_tables([a, b])
    assert len(tables) == 1  # all BinZTracers are written to a single table

    d = BinZTracer.from_tables(tables)
    assert len(d) == 2  # this list of tables recovers both BinZTracers
    assert d["fred"] == a
    assert d["wilma"] == b


def test_make_binrichness_tracer():
    tracer = BinRichnessTracer.make(
        "bin_richness",
        name="barney",
        richness_lower=0.25,
        richness_upper=1.0,
    )
    assert isinstance(tracer, BinRichnessTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "barney"
    assert tracer.richness_upper == 1.0
    assert tracer.richness_lower == 0.25


def test_binrichnesstracer_tables():
    a = BinRichnessTracer.make(
        "bin_richness", name="barney", richness_lower=0.0, richness_upper=0.5
    )
    b = BinRichnessTracer.make(
        "bin_richness", name="betty", richness_lower=1.25, richness_upper=2.0
    )
    tables = BinRichnessTracer.to_tables([a, b])
    assert len(tables) == 1
    d = BinRichnessTracer.from_tables(tables)
    assert len(d) == 2  # this list of tables recovers both BinRichnessTracers
    assert d["barney"] == a
    assert d["betty"] == b


def test_make_binradiustracer():
    tracer = BinRadiusTracer.make(
        "bin_radius", name="pebbles", r_lower=1.0, r_center=2.0, r_upper=3.0
    )
    assert isinstance(tracer, BinRadiusTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "pebbles"
    assert tracer.r_lower == 1.0
    assert tracer.r_center == 2.0
    assert tracer.r_upper == 3.0


def test_binradiustracer_tables():
    a = BinRadiusTracer.make(
        "bin_radius", name="pebbles", r_lower=1.0, r_center=2.0, r_upper=3.0
    )
    b = BinRadiusTracer.make(
        "bin_radius", name="bambam", r_lower=3.0, r_center=4.0, r_upper=5.0
    )
    tables = BinRadiusTracer.to_tables([a, b])
    assert len(tables) == 1
    d = BinRadiusTracer.from_tables(tables)
    assert len(d) == 2
    assert d["pebbles"] == a
    assert d["bambam"] == b
