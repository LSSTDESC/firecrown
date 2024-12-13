"""
Tests for the module firecrown.likelihood.number_counts_magnification_bias_systematic.
"""

import numpy as np
import numpy.typing as npt

import firecrown.likelihood.number_counts as nc
import firecrown.parameters as fp
import firecrown.modeling_tools as mt


def test_creation_nc_magnification_bias_systematic():
    sys = nc.MagnificationBiasSystematic(sacc_tracer="lens0")
    assert sys.parameter_prefix == "lens0"
    assert sys.eta is None
    assert sys.r_lim is None
    assert sys.sig_c is None
    assert sys.z_c is None
    assert sys.z_m is None


def test_update_nc_magnification_bias_systematic(
    tools_with_vanilla_cosmology: mt.ModelingTools,
):
    from sys import settrace

    # some global context to be used in the tracing. We are relying on
    # 'trace_call' to act as a closure that captures these names.
    f = open("trace.log", "w")  # the file used for logging
    level = 0  # the call nesting level
    entry = 0  # sequential entry number for each record

    print("entry\tevent\tlevel\tfunction\tvalue\textra", file=f)

    def trace_call(fr, ev, arg):
        nonlocal level
        nonlocal entry
        code = fr.f_code
        extra = ""
        match ev:
            case "call":
                entry += 1
                level += 1
                nargs = code.co_argcount
                # slice the tuple to get only argument names
                argnames = code.co_varnames[:nargs]
                if nargs > 0 and code.co_varnames[0] == "self":
                    val = fr.f_locals["self"]
                    extra = f"{type(val).__name__}"
                print(
                    f"{entry}\tcall\t{level}\t{code.co_qualname}\t{argnames}\t{extra}",
                    file=f,
                )
            case "return":
                entry += 1
                extra = f"{type(arg).__name__}"
                print(
                    f"{entry}\treturn\t{level}\t{code.co_qualname}\t{arg}\t{extra}",
                    file=f,
                )
                level -= 1
            case "exception":
                entry += 1
                print(
                    f"{entry}\texception\t{level}\t{code.co_qualname}\t\t{extra}",
                    file=f,
                )
        return trace_call

    settrace(trace_call)

    # Now we have the read test...
    sys = nc.MagnificationBiasSystematic(sacc_tracer="lens0")
    assert sys.parameter_prefix == "lens0"
    sys.update(
        fp.ParamsMap(
            {
                "lens0_eta": 19.0,
                "lens0_r_lim": 24.0,
                "lens0_sig_c": 9.83,
                "lens0_z_c": 0.39,
                "lens0_z_m": 0.055,
            }
        )
    )
    assert sys.eta == 19.0
    assert sys.r_lim == 24.0
    assert sys.sig_c == 9.83
    assert sys.z_c == 0.39
    assert sys.z_m == 0.055

    ta = nc.NumberCountsArgs(
        z=np.array([1.0, 2.0]),
        dndz=np.array([1.0, 0.5]),
    )
    new_z: npt.NDArray[np.float64]
    new_mag_bias: npt.NDArray[np.float64]
    new_ta = sys.apply(tools_with_vanilla_cosmology, ta)
    assert new_ta.mag_bias is not None
    new_z, new_mag_bias = new_ta.mag_bias
    expected_zs: npt.NDArray[np.float64] = np.array([1.0, 2.0])
    expected_mag_biases: npt.NDArray[np.float64] = np.array([0.53728088, 1.22697163])

    assert np.allclose(new_z, expected_zs)
    assert np.allclose(new_mag_bias, expected_mag_biases)
    settrace(None)
    f.close()
