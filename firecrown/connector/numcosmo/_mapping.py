"""Support of Firecrown likelihood use in NumCosmo.

The subpackages and modules in this package depend upon NumCosmo, and can not
be used without an installation of NumCosmo.
"""

import warnings
from typing import cast

import numpy as np
from numcosmo_py import GObject, Nc, Ncm, dict_to_var_dict, var_dict_to_dict

from firecrown.connector.mapping import Mapping, build_ccl_background_dict
from firecrown.connector.numcosmo import helpers
from firecrown.likelihood import NamedParameters
from firecrown.modeling_tools import (
    CCLCalculatorArgs,
    PowerSpec,
    PoweSpecAmplitudeParameter,
)


def _var_dict_to_named_parameters(value: None | Ncm.VarDict) -> NamedParameters:
    """Convert a NumCosmo variable dictionary to Firecrown parameters."""
    parameters = NamedParameters()
    if value is not None:
        parameters.set_from_basic_dict(var_dict_to_dict(value))
    return parameters


def _named_parameters_to_var_dict(
    value: None | NamedParameters,
) -> None | Ncm.VarDict:
    """Convert Firecrown parameters to a NumCosmo variable dictionary."""
    if value is None:
        return None
    return dict_to_var_dict(value.convert_to_basic_dict())


def _extract_pk(
    power_spec: Nc.PowspecML | Nc.PowspecMNL,
    mapping: Mapping,
    hi_cosmo: Nc.HICosmo,
) -> PowerSpec:
    """Sample a NumCosmo power spectrum object into a Firecrown PowerSpec.

    Shared by :meth:`MappingNumCosmo.extract_pk_linear` and
    :meth:`MappingNumCosmo.extract_pk_nonlinear`; callers are responsible for
    verifying that `power_spec` is not None before calling this function.

    :param power_spec: the NumCosmo power spectrum object to sample
    :param mapping: the mapping object used to convert redshift to scale factor
    :param hi_cosmo: the NumCosmo HICosmo object containing cosmological parameters
    :returns: a dictionary containing the power spectrum with scale factors, wave
             numbers, and power spectrum values
    """
    spline = power_spec.get_spline_2d(hi_cosmo)
    z = np.array(spline.peek_xv().dup_array())
    k = np.array(spline.peek_yv().dup_array())
    scale = mapping.redshift_to_scale_factor(z)
    p_k = np.transpose(np.array(spline.peek_zm().dup_array()).reshape(len(k), len(z)))
    p_k = mapping.redshift_to_scale_factor_p_k(p_k)
    return {"a": scale, "k": k, "delta_matter:delta_matter": p_k}


class MappingNumCosmo(GObject.Object):
    """Mapping support for NumCosmo.

    This is a subclass of :class:`Mapping` that provides a mapping from a NumCosmo
    Cosmological model to a CCL cosmology. It also converts NumCosmo models to
    :class:`ParamsMap` objects.
    """

    __gtype_name__ = "FirecrownMappingNumCosmo"

    def __init__(
        self,
        require_nonlinear_pk: None | bool = None,
        p_ml: None | Nc.PowspecML = None,
        p_mnl: None | Nc.PowspecMNL = None,
        dist: None | Nc.Distance = None,
    ) -> None:
        """Initialize a MappingNumCosmo object.

        :param p_ml: optional PowspecML object
        :param p_mnl: optional PowspecMNL object
        :param dist: optional Distance object
        """
        super().__init__(p_ml=p_ml, p_mnl=p_mnl, dist=dist)
        if p_mnl is not None:
            assert (
                p_ml is not None
            ), "PowspecML object must be provided when using PowspecMNL."
        self.mapping: Mapping
        self._mapping_name: str
        self._p: None | helpers.PowerSpec
        self._dist: Nc.Distance

        if require_nonlinear_pk is not None:
            warnings.warn(
                "The require_nonlinear_pk argument is deprecated and will be removed "
                "in future versions. This configuration is now handled by the "
                "likelihood factory function.",
                DeprecationWarning,
                stacklevel=2,
            )

        if p_ml:
            if p_mnl:
                self._p = helpers.PowerSpec(p_ml, p_mnl)
            else:
                self._p = helpers.PowerSpec(p_ml, None)
        else:
            self._p = None

    def _get_mapping_name(self) -> str:
        """Return the mapping name.

        :returns: the name of the mapping.
        """
        return self._mapping_name

    def _set_mapping_name(self, value: str) -> None:
        """Set the mapping name.

        This method also sets the :attr:`mapping` property to a default-
        initialized :class:`Mapping` object.

        :param value: the new name of the mapping
        """
        self._mapping_name = value
        self.mapping = Mapping()

    # cast: gi-stubs types GObject.Property as "Property", not the property's
    # actual value type; getter/setter above guarantee this is really a str.
    mapping_name: str = cast(
        str,
        GObject.Property(
            type=str,
            default="default",
            flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT_ONLY,
            getter=_get_mapping_name,
            setter=_set_mapping_name,
        ),
    )

    def _get_p_ml(self) -> None | Nc.PowspecML:
        """Return the NumCosmo PowspecML object.

        :param value: the NumCosmo PowspecML object, or None
        """
        if self._p is None:
            return None
        return self._p.linear

    def _set_p_ml(self, value: None | Nc.PowspecML) -> None:
        """Set the NumCosmo PowspecML object.

        :param value: the new value to be set
        """
        if value is None:
            self._p = None
        else:
            self._p = helpers.PowerSpec(value, None)

    # cast: gi-stubs types GObject.Property as "Property", not the property's
    # actual value type; getter/setter above guarantee this is really a PowspecML.
    p_ml: Nc.PowspecML = cast(
        Nc.PowspecML,
        GObject.Property(
            type=Nc.PowspecML,
            flags=GObject.ParamFlags.READWRITE,
            getter=_get_p_ml,
            setter=_set_p_ml,
        ),
    )

    def _get_p_mnl(self) -> None | Nc.PowspecMNL:
        """Return the NumCosmo PowspecMNL object.

        :returns: the NumCosmo PowspecMNL object, or None
        """
        if self._p is None:
            return None
        return self._p.nonlinear

    def _set_p_mnl(self, value: None | Nc.PowspecMNL) -> None:
        """Set the NumCosmo PowspecMNL object.

        It is illegal to set a PowspecMNL object when there is no PowspecML
        object. Currently, we can not raise an exception because it does
        not propagate through the C interface between this code and the
        Python code that calls it.

        :param value: the new value to be set
        """
        if self._p is not None:
            self._p.nonlinear = value

    # cast: gi-stubs types GObject.Property as "Property", not the property's
    # actual value type; getter/setter above guarantee this is really a PowspecMNL.
    p_mnl: Nc.PowspecMNL = cast(
        Nc.PowspecMNL,
        GObject.Property(
            type=Nc.PowspecMNL,
            flags=GObject.ParamFlags.READWRITE,
            getter=_get_p_mnl,
            setter=_set_p_mnl,
        ),
    )

    def _get_dist(self) -> None | Nc.Distance:
        """Return the NumCosmo Distance object.

        :returns: the NumCosmo Distance object, or None
        """
        return self._dist

    def _set_dist(self, value: Nc.Distance) -> None:
        """Set the NumCosmo Distance object."""
        self._dist = value

    # cast: gi-stubs types GObject.Property as "Property", not the property's
    # actual value type; getter/setter above guarantee this is really a Distance.
    dist: Nc.Distance = cast(
        Nc.Distance,
        GObject.Property(
            type=Nc.Distance,
            flags=GObject.ParamFlags.READWRITE,
            getter=_get_dist,
            setter=_set_dist,
        ),
    )

    def set_params_from_numcosmo(
        self, mset: Ncm.MSet, amplitude_parameter: PoweSpecAmplitudeParameter
    ) -> None:  # pylint: disable-msg=too-many-locals
        """Set the parameters of the contained Mapping object.

        :param mset: the NumCosmo MSet object from which to get the parameters
        """
        hi_cosmo = mset.peek(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert isinstance(hi_cosmo, Nc.HICosmo)

        if self._p is not None:
            self._p.prepare_if_needed(hi_cosmo)
        self._dist.prepare_if_needed(hi_cosmo)

        Omega_b = hi_cosmo.Omega_b0()
        Omega_c = hi_cosmo.Omega_c0()
        Omega_k = hi_cosmo.Omega_k0()
        Neff = hi_cosmo.Neff()
        T_gamma0 = hi_cosmo.T_gamma0()

        m_nu: float | list[float] = 0.0
        if hi_cosmo.NMassNu() > 0:
            m_nu = [hi_cosmo.MassNuInfo(i)[0] for i in range(hi_cosmo.NMassNu())]

        match hi_cosmo:
            case Nc.HICosmoDEXcdm():
                w0 = hi_cosmo.props.w
                wa = 0.0
            case Nc.HICosmoDECpl():
                w0 = hi_cosmo.props.w0
                wa = hi_cosmo.props.w1
            case _:
                raise ValueError(f"NumCosmo object {type(hi_cosmo)} not supported.")

        A_s, sigma8 = helpers.get_amplitude_parameters(
            amplitude_parameter, self.p_ml, hi_cosmo
        )

        # pylint: disable=duplicate-code
        self.mapping.set_params(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=hi_cosmo.h(),
            A_s=A_s,
            sigma8=sigma8,
            n_s=helpers.get_hiprim(hi_cosmo).props.n_SA,
            Omega_k=Omega_k,
            Neff=Neff,
            m_nu=m_nu,
            w0=w0,
            wa=wa,
            T_CMB=T_gamma0,
        )
        # pylint: enable=duplicate-code

    def calculate_ccl_args(  # pylint: disable-msg=too-many-locals
        self, mset: Ncm.MSet
    ) -> CCLCalculatorArgs:
        """Calculate the arguments necessary for CCL for this sample.

        :param mset: the NumCosmo MSet object from which to get the parameters
        :returns: a dictionary of the arguments required by CCL
        """
        hi_cosmo = mset.peek(Nc.HICosmo.id())  # pylint: disable=no-value-for-parameter
        assert isinstance(hi_cosmo, Nc.HICosmo)

        d_spline = self._dist.comoving_distance_spline.peek_spline()
        z_dist = np.array(d_spline.get_xv().dup_array())
        c_dist = np.array(d_spline.get_yv().dup_array())

        chi = (np.flip(c_dist) * hi_cosmo.RH_Mpc()).astype(np.float64)
        scale_distances = self.mapping.redshift_to_scale_factor(z_dist)
        h_over_h0 = np.array(
            [hi_cosmo.E(z) for z in reversed(z_dist)], dtype=np.float64
        )

        # Too many points in the redshift spline can result in scale factors
        # that are too close together for CCL to handle. This checks for
        # duplicate scale factors and removes them.
        a_unique, a_unique_indices = np.unique(scale_distances, return_index=True)
        scale_distances = a_unique
        chi = chi[a_unique_indices]
        h_over_h0 = h_over_h0[a_unique_indices]

        ccl_args: CCLCalculatorArgs = {
            "background": build_ccl_background_dict(
                a=scale_distances, chi=chi, h_over_h0=h_over_h0
            )
        }

        if self._p:
            ccl_args["pk_linear"] = self.extract_pk_linear(self.mapping, hi_cosmo)
            if self._p.nonlinear:
                ccl_args["pk_nonlin"] = self.extract_pk_nonlinear(
                    self.mapping, hi_cosmo
                )
        return ccl_args

    def extract_pk_nonlinear(self, mapping: Mapping, hi_cosmo: Nc.HICosmo) -> PowerSpec:
        """Extract the nonlinear power spectrum from the NumCosmo PowspecMNL object.

        :param mapping: The mapping object used to convert redshift to scale factor.
        :param hi_cosmo: The NumCosmo HICosmo object containing cosmological parameters.
        :returns: A dictionary containing the nonlinear power spectrum with scale
                 factors, wave numbers, and power spectrum values.
        """
        assert self._p is not None
        assert self._p.nonlinear is not None
        return _extract_pk(self._p.nonlinear, mapping, hi_cosmo)

    def extract_pk_linear(self, mapping: Mapping, hi_cosmo: Nc.HICosmo) -> PowerSpec:
        """Extract the linear power spectrum from the NumCosmo PowspecMNL object.

        :param mapping: The mapping object used to convert redshift to scale factor.
        :param hi_cosmo: The NumCosmo HICosmo object containing cosmological parameters.
        :returns: A dictionary containing the linear power spectrum with scale factors,
                 wave numbers, and power spectrum values.
        """
        assert self._p is not None
        return _extract_pk(self._p.linear, mapping, hi_cosmo)
