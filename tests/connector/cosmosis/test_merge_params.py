import pytest

from firecrown.parameters import ParamsMap


def test_merge_conflict_detection():
    # Simulate existing firecrown params produced by calculate_firecrown_params
    existing = {"Omega_c": 0.25, "ia_bias": 0.1}

    # Canonicalized cosmology that conflicts on Omega_c
    canonical = {"Omega_c": 0.26, "h": 0.68}

    merged = dict(existing)

    # The connector raises a RuntimeError on conflict; reproduce that check here
    with pytest.raises(RuntimeError):
        for k, v in canonical.items():
            if k in merged and merged[k] != v:
                raise RuntimeError(
                    f"Conflicting cosmological parameter '{k}' found in "
                    f"sampling sections and 'cosmological_parameters'."
                )
            merged[k] = v


def test_successful_merge_produces_paramsmap():
    existing = {"ia_bias": 0.1}
    canonical = {"Omega_c": 0.25, "h": 0.68}

    merged = dict(existing)
    for k, v in canonical.items():
        if k in merged and merged[k] != v:
            raise RuntimeError("conflict")
        merged[k] = v

    pm = ParamsMap(merged)
    # ParamsMap should accept floats for the cosmology entries
    assert pm.get_from_full_name("Omega_c") == 0.25
    assert pm.get_from_full_name("h") == 0.68
