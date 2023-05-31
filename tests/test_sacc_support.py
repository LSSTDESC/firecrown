"""
Tests for function supporting SACC.

"""
from firecrown.sacc_support import (
    BinZTracer,
    BinRichnessTracer,
    BinRadiusTracer,
    ClusterSurveyTracer,
)


def test_make_binztracer():
    tracer = BinZTracer.make("bin_z", name="fred", z_lower=0.5, z_upper=1.0)
    assert isinstance(tracer, BinZTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "fred"
    assert tracer.z_lower == 0.5
    assert tracer.z_upper == 1.0


def test_binztracer_equality():
    a = BinZTracer.make("bin_z", name="fred", z_lower=0.5, z_upper=1.0)
    b = BinZTracer.make("bin_z", name="fred", z_lower=0.5, z_upper=1.0)
    c = BinZTracer.make("bin_z", name="wilma", z_lower=0.5, z_upper=1.0)
    d = BinZTracer.make("bin_z", name="fred", z_lower=0.6, z_upper=1.0)
    e = BinZTracer.make("bin_z", name="fred", z_lower=0.5, z_upper=1.1)
    assert a == b
    assert a != "fred"
    assert a != c
    assert a != d
    assert a != e


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


def test_binrichnesstracer_equality():
    a = BinRichnessTracer.make(
        "bin_richness", name="fred", richness_lower=0.5, richness_upper=1.0
    )
    b = BinRichnessTracer.make(
        "bin_richness", name="fred", richness_lower=0.5, richness_upper=1.0
    )
    c = BinRichnessTracer.make(
        "bin_richness", name="wilma", richness_lower=0.5, richness_upper=1.0
    )
    d = BinRichnessTracer.make(
        "bin_richness", name="fred", richness_lower=0.6, richness_upper=1.0
    )
    e = BinRichnessTracer.make(
        "bin_richness", name="fred", richness_lower=0.5, richness_upper=1.1
    )
    assert a == b
    assert a != "fred"
    assert a != c
    assert a != d
    assert a != e


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


def test_binradiustracer_equality():
    a = BinRadiusTracer.make(
        "bin_radius", name="fred", r_lower=0.5, r_center=0.75, r_upper=1.0
    )
    b = BinRadiusTracer.make(
        "bin_radius", name="fred", r_lower=0.5, r_center=0.75, r_upper=1.0
    )
    c = BinRadiusTracer.make(
        "bin_radius", name="wilma", r_lower=0.5, r_center=0.75, r_upper=1.0
    )
    d = BinRadiusTracer.make(
        "bin_radius", name="fred", r_lower=0.6, r_center=0.75, r_upper=1.0
    )
    e = BinRadiusTracer.make(
        "bin_radius", name="fred", r_lower=0.5, r_center=0.8, r_upper=1.0
    )
    f = BinRadiusTracer.make(
        "bin_radius", name="fred", r_lower=0.5, r_center=0.75, r_upper=1.1
    )
    assert a == b
    assert a != "fred"
    assert a != c
    assert a != d
    assert a != e
    assert a != f


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


def test_make_clustersurveytracer():
    tracer = ClusterSurveyTracer.make("cluster_survey", name="bullwinkle", sky_area=1.0)
    assert isinstance(tracer, ClusterSurveyTracer)
    assert tracer.quantity == "generic"
    assert tracer.name == "bullwinkle"
    assert tracer.sky_area == 1.0


def test_clustersurveytracer_equality():
    a = ClusterSurveyTracer.make("cluster_survey", name="bullwinkle", sky_area=1.0)
    b = ClusterSurveyTracer.make("cluster_survey", name="bullwinkle", sky_area=1.0)
    c = ClusterSurveyTracer.make("cluster_survey", name="rocky", sky_area=1.0)
    d = ClusterSurveyTracer.make("cluster_survey", name="boris", sky_area=2.0)

    assert a == b
    assert a != "bullwinkle"
    assert a != c
    assert a != d


def test_clustersurveytracer_tables():
    a = ClusterSurveyTracer.make("cluster_survey", name="bullwinkle", sky_area=1.0)
    b = ClusterSurveyTracer.make("cluster_survey", name="rocky", sky_area=2.0)
    tables = ClusterSurveyTracer.to_tables([a, b])
    assert len(tables) == 1
    d = ClusterSurveyTracer.from_tables(tables)
    assert len(d) == 2
    assert d["bullwinkle"] == a
    assert d["rocky"] == b
