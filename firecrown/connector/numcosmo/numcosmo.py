"""Support of Firecrown likelihood use in NumCosmo.

The subpackages and modules in this package depend upon NumCosmo, and can not
be used without an installation of NumCosmo.
"""

import warnings

import numpy as np
from typing_extensions import assert_never

from numcosmo_py import Nc, Ncm, GObject, var_dict_to_dict, dict_to_var_dict

from firecrown.likelihood.likelihood import load_likelihood
from firecrown.likelihood.likelihood import Likelihood
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from firecrown.connector.mapping import Mapping, build_ccl_background_dict
from firecrown.modeling_tools import ModelingTools
from firecrown.ccl_factory import (
    CCLCalculatorArgs,
    PowerSpec,
    CCLFactory,
    PoweSpecAmplitudeParameter,
    CCLCreationMode,
)


def get_hiprim(hi_cosmo: Nc.HICosmo) -> Nc.HIPrimPowerLaw:
    """Return the HIPrim object from a NumCosmo HICosmo object.

    If hi_cosmo does not have a HIPrim object, a ValueError is raised.
    If the HIPrim object is not of type HIPrimPowerLaw, a ValueError is raised.

    :param hi_cosmo: NumCosmo HICosmo object
    :return: the HIPrim object contained in hi_cosmo
    """
    hiprim = hi_cosmo.peek_submodel_by_mid(Nc.HIPrim.id())
    if not hiprim:
        raise ValueError("NumCosmo object must include a HIPrim object.")
    if not isinstance(hiprim, Nc.HIPrimPowerLaw):
        raise ValueError(f"NumCosmo HIPrim object type {type(hiprim)} not supported.")
    return hiprim


def get_amplitude_parameters(
    ccl_factory: CCLFactory,
    p_ml: None | Nc.PowspecML,
    hi_cosmo: Nc.HICosmo,
) -> tuple[float | None, float | None]:
    """
    Calculate the amplitude parameters for CCL.

    :param ccl_factory: the CCL factory object
    :param p_ml: the NumCosmo PowspecML object, or None
    :param hi_cosmo: the NumCosmo cosmology object
    :return: a tuple of the amplitude parameters, (A_s, sigma8), with only one set.
    """
    A_s: float | None = None
    sigma8: float | None = None

    # mypy verifies that the match statement below is exhaustive
    match ccl_factory.amplitude_parameter:
        case PoweSpecAmplitudeParameter.SIGMA8:
            if p_ml is None:
                raise ValueError("PowspecML object must be provided when using sigma8.")
            sigma8 = p_ml.sigma_tophat_R(hi_cosmo, 1.0e-7, 0.0, 8.0 / hi_cosmo.h())
        case PoweSpecAmplitudeParameter.AS:
            A_s = get_hiprim(hi_cosmo).SA_Ampl()
        case _ as unreachable:
            assert_never(unreachable)
    assert A_s is not None or sigma8 is not None
    return A_s, sigma8


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
        super().__init__(p_ml=p_ml, p_mnl=p_mnl, dist=dist)  # type: ignore
        self.mapping: Mapping
        self._mapping_name: str
        self._p_ml: None | Nc.PowspecML
        self._p_mnl: None | Nc.PowspecMNL
        self._dist: Nc.Distance

        if require_nonlinear_pk is not None:
            warnings.warn(
                "The require_nonlinear_pk argument is deprecated and will be removed "
                "in future versions. This configuration is now handled by the "
                "likelihood factory function.",
                DeprecationWarning,
                stacklevel=2,
            )

        if not hasattr(self, "_p_ml"):
            self._p_ml = None

        if not hasattr(self, "_p_mnl"):
            self._p_mnl = None

    def _get_mapping_name(self) -> str:
        """Return the mapping name.

        :return: the name of the mapping.
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

    mapping_name = GObject.Property(
        type=str,
        default="default",
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT_ONLY,
        getter=_get_mapping_name,
        setter=_set_mapping_name,
    )

    def _get_p_ml(self) -> None | Nc.PowspecML:
        """Return the NumCosmo PowspecML object.

        :param value: the NumCosmo PowspecML object, or None
        """
        return self._p_ml

    def _set_p_ml(self, value: None | Nc.PowspecML) -> None:
        """Set the NumCosmo PowspecML object.

        :param value: the new value to be set
        """
        self._p_ml = value

    p_ml = GObject.Property(
        type=Nc.PowspecML,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_p_ml,
        setter=_set_p_ml,
    )

    def _get_p_mnl(self) -> None | Nc.PowspecMNL:
        """Return the NumCosmo PowspecMNL object.

        :return: the NumCosmo PowspecMNL object, or None
        """
        return self._p_mnl

    def _set_p_mnl(self, value: None | Nc.PowspecMNL) -> None:
        """Set the NumCosmo PowspecMNL object.

        :param value: the new value to be set
        """
        self._p_mnl = value

    p_mnl = GObject.Property(
        type=Nc.PowspecMNL,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_p_mnl,
        setter=_set_p_mnl,
    )

    def _get_dist(self) -> None | Nc.Distance:
        """Return the NumCosmo Distance object.

        :return: the NumCosmo Distance object, or None
        """
        return self._dist

    def _set_dist(self, value: Nc.Distance) -> None:
        """Set the NumCosmo Distance object."""
        self._dist = value

    dist = GObject.Property(
        type=Nc.Distance,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_dist,
        setter=_set_dist,
    )

    def set_params_from_numcosmo(
        self, mset: Ncm.MSet, ccl_factory: CCLFactory
    ) -> None:  # pylint: disable-msg=too-many-locals
        """Set the parameters of the contained Mapping object.

        :param mset: the NumCosmo MSet object from which to get the parameters
        """
        hi_cosmo = mset.peek(Nc.HICosmo.id())
        assert isinstance(hi_cosmo, Nc.HICosmo)

        if self._p_ml is not None:
            self._p_ml.prepare_if_needed(hi_cosmo)
        if self._p_mnl is not None:
            self._p_mnl.prepare_if_needed(hi_cosmo)
        self._dist.prepare_if_needed(hi_cosmo)

        Omega_b = hi_cosmo.Omega_b0()
        Omega_c = hi_cosmo.Omega_c0()
        Omega_k = hi_cosmo.Omega_k0()
        Neff = hi_cosmo.Neff()
        T_gamma0 = hi_cosmo.T_gamma0()

        m_nu: float | list[float] = 0.0
        if hi_cosmo.NMassNu() > 0:
            assert hi_cosmo.NMassNu() <= 3
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

        A_s, sigma8 = get_amplitude_parameters(ccl_factory, self._p_ml, hi_cosmo)

        assert (A_s is not None) or (sigma8 is not None)

        # pylint: disable=duplicate-code
        self.mapping.set_params(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=hi_cosmo.h(),
            A_s=A_s,
            sigma8=sigma8,
            n_s=get_hiprim(hi_cosmo).props.n_SA,
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
        :return: a dictionary of the arguments required by CCL
        """
        pk_linear: None | PowerSpec = None
        pk_nonlin: None | PowerSpec = None
        hi_cosmo = mset.peek(Nc.HICosmo.id())
        assert isinstance(hi_cosmo, Nc.HICosmo)

        if self._p_ml:
            p_m_spline = self._p_ml.get_spline_2d(hi_cosmo)
            z = np.array(p_m_spline.peek_xv().dup_array())
            k = np.array(p_m_spline.peek_yv().dup_array())

            scale = self.mapping.redshift_to_scale_factor(z)
            p_k = np.transpose(
                np.array(p_m_spline.peek_zm().dup_array()).reshape(len(k), len(z))
            )
            p_k = self.mapping.redshift_to_scale_factor_p_k(p_k)
            pk_linear = {
                "a": scale,
                "k": k,
                "delta_matter:delta_matter": p_k,
            }

        if self._p_mnl:
            p_mnl_spline = self._p_mnl.get_spline_2d(hi_cosmo)
            z = np.array(p_mnl_spline.peek_xv().dup_array())
            k = np.array(p_mnl_spline.peek_yv().dup_array())

            scale_mpnl = self.mapping.redshift_to_scale_factor(z)
            p_mnl = np.transpose(
                np.array(p_mnl_spline.peek_zm().dup_array()).reshape(len(k), len(z))
            )
            p_mnl = self.mapping.redshift_to_scale_factor_p_k(p_mnl)
            pk_nonlin = {
                "a": scale_mpnl,
                "k": k,
                "delta_matter:delta_matter": p_mnl,
            }

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
        if pk_linear:
            ccl_args["pk_linear"] = pk_linear
        if pk_nonlin:
            ccl_args["pk_nonlin"] = pk_nonlin

        return ccl_args

    def create_params_map(self, model_list: list[str], mset: Ncm.MSet) -> ParamsMap:
        """Create a ParamsMap from a NumCosmo MSet.

        All the models named in model_list must be in the model set `mset`, or a
        RuntimeError will be raised.

        :param model_list: list of model names
        :param mset: the NumCosmo MSet object from which to get the parameters
        :return: a ParamsMap containing the parameters of the models in model_list
        """
        params_map = ParamsMap()
        for model_ns in model_list:
            mid = mset.get_id_by_ns(model_ns)
            if mid < 0:
                raise RuntimeError(
                    f"Model name {model_ns} was not found in the model set."
                )
            model = mset.peek(mid)
            # Since we have already verified that the model name exists in the
            # model set, if the model is not found we have encountered an
            # unrecoverable error.
            assert model is not None

            param_names = model.param_names()
            model_dict = {
                param: model.param_get_by_name(param) for param in param_names
            }
            params_map = self._update_params_map(model_list, params_map, model_dict)

        params_map = self._update_params_map(
            model_list, params_map, self.mapping.asdict()
        )

        return params_map

    def _update_params_map(self, model_list, params_map, model_dict):
        shared_keys = set(model_dict).intersection(params_map)
        if len(shared_keys) > 0:
            raise RuntimeError(
                f"The following keys `{shared_keys}` appear in more than one model "
                f"used by the module {model_list} or cosmological parameters."
            )
        params_map = ParamsMap({**params_map, **model_dict})
        return params_map


class NumCosmoData(Ncm.Data):
    """NumCosmoData is a subclass of Ncm.Data.

    This subclass also implements NumCosmo likelihood object virtual methods using
    the prefix `do_`. This class implements a general likelihood.
    """

    __gtype_name__ = "FirecrownNumCosmoData"

    def __init__(self):
        """Initialize a NumCosmoData object.

        Default values are provided for all attributes; most default are None.
        """
        super().__init__()
        self.likelihood: Likelihood
        self.tools: ModelingTools
        self._model_list: list[str]
        self._nc_mapping: MappingNumCosmo
        self._likelihood_source: None | str = None
        self._likelihood_build_parameters: None | NamedParameters = None
        self._starting_deserialization: bool = False
        self.dof: int = 100
        self.len: int = 100
        self.set_init(True)

    def _get_model_list(self) -> list[str]:
        """Return the list of model names.

        :return: the names of the contained models
        """
        return self._model_list

    def _set_model_list(self, value: list[str]) -> None:
        """Set the list of model names.

        :param value: the new of model names
        """
        self._model_list = value

    model_list = GObject.Property(
        type=GObject.TYPE_STRV,  # type: ignore
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_model_list,
        setter=_set_model_list,
    )

    def _get_nc_mapping(self) -> MappingNumCosmo:
        """Return the MappingNumCosmo object.

        :return: the MappingNumCosmo object
        """
        return self._nc_mapping

    def _set_nc_mapping(self, value: MappingNumCosmo) -> None:
        """Set the MappingNumCosmo object.

        :param: the new value for the MappingNumCosmo object
        """
        self._nc_mapping = value

    nc_mapping = GObject.Property(
        type=MappingNumCosmo,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_nc_mapping,
        setter=_set_nc_mapping,
    )

    def _set_likelihood_from_factory(self) -> None:
        """Deserialize the likelihood."""
        assert self._likelihood_source is not None
        assert self._likelihood_build_parameters is not None
        likelihood, tools = load_likelihood(
            self._likelihood_source, self._likelihood_build_parameters
        )
        assert isinstance(likelihood, Likelihood)
        assert isinstance(tools, ModelingTools)
        self.likelihood = likelihood
        self.tools = tools

    def _get_likelihood_source(self) -> None | str:
        """Return the likelihood string defining the factory function.

        :return: the filename of the likelihood factory function
        """
        return self._likelihood_source

    def _set_likelihood_source(self, value: None | str):
        """Set the likelihood string defining the factory function.

        :param value: the filename of the likelihood factory function
        """
        if value is not None:
            self._likelihood_source = value
            if self._starting_deserialization:
                self._set_likelihood_from_factory()
                self._starting_deserialization = False
            else:
                self._starting_deserialization = True

    likelihood_source = GObject.Property(
        type=str,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_likelihood_source,
        setter=_set_likelihood_source,
    )

    def _get_likelihood_build_parameters(self) -> None | Ncm.VarDict:
        """Return the likelihood build parameters."""
        if self._likelihood_build_parameters is None:
            return None
        return dict_to_var_dict(
            self._likelihood_build_parameters.convert_to_basic_dict()
        )

    def _set_likelihood_build_parameters(self, value: None | Ncm.VarDict) -> None:
        """Set the likelihood build parameters.

        :param value: the parameters used to build the likelihood
        """
        self._likelihood_build_parameters = NamedParameters()
        if value is not None:
            self._likelihood_build_parameters.set_from_basic_dict(
                var_dict_to_dict(value)
            )

        if self._starting_deserialization:
            self._set_likelihood_from_factory()
            self._starting_deserialization = False
        else:
            self._starting_deserialization = True

    likelihood_build_parameters = GObject.Property(
        type=Ncm.VarDict,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_likelihood_build_parameters,
        setter=_set_likelihood_build_parameters,
    )

    @classmethod
    def new_from_likelihood(
        cls,
        likelihood: Likelihood,
        model_list: list[str],
        tools: ModelingTools,
        nc_mapping: MappingNumCosmo,
        likelihood_source: None | str = None,
        likelihood_build_parameters: None | NamedParameters = None,
    ) -> "NumCosmoData":
        """Initialize a NumCosmoGaussCov object.

        This object represents a Gaussian likelihood with a constant covariance.

        :param likelihood: the likelihood object
        :param model_list: the list of model names
        :param tools: the modeling tools
        :param nc_mapping: the mapping object
        :param likelihood_source: the filename for the likelihood factory function
        :param likelihood_build_parameters: the build parameters of the likelihood
        """
        nc_data: NumCosmoData = GObject.new(
            cls,
            model_list=model_list,
            nc_mapping=nc_mapping,
        )

        nc_data.likelihood = likelihood
        nc_data.tools = tools
        # pylint: disable=protected-access
        nc_data._likelihood_source = likelihood_source
        nc_data._likelihood_build_parameters = likelihood_build_parameters
        # pylint: enable=protected-access

        return nc_data

    def do_get_length(self) -> int:  # pylint: disable-msg=arguments-differ
        """Implements the virtual Ncm.Data method get_length.

        :return: the number of data points in the likelihood
        """
        return self.len

    def do_get_dof(self) -> int:  # pylint: disable-msg=arguments-differ
        """Implements the virtual Ncm.Data method get_dof.

        :return: the number of degrees of freedom in the likelihood
        """
        return self.dof

    def do_begin(self) -> None:  # pylint: disable-msg=arguments-differ
        """Implements the virtual Ncm.Data method `begin`.

        This method usually do some groundwork in the data
        before the actual calculations. For example, if the likelihood
        involves the decomposition of a constant matrix, it can be done
        during `begin` once and then used afterwards.
        """

    def do_prepare(  # pylint: disable-msg=arguments-differ
        self, mset: Ncm.MSet
    ) -> None:
        """Implements the virtual method Ncm.Data `prepare`.

        This method should do all the necessary calculations using mset
        to be able to calculate the likelihood afterwards.

        :param mset: the model set
        """
        self.dof = self.len - mset.fparams_len()
        self.likelihood.reset()
        self.tools.reset()

        self._nc_mapping.set_params_from_numcosmo(mset, self.tools.ccl_factory)
        params_map = self._nc_mapping.create_params_map(self.model_list, mset)

        self.likelihood.update(params_map)
        self.tools.update(params_map)
        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            self.tools.prepare(
                calculator_args=self._nc_mapping.calculate_ccl_args(mset)
            )
        else:
            self.tools.prepare()

    def do_m2lnL_val(self, _) -> float:  # pylint: disable-msg=arguments-differ
        """Implements the virtual method `m2lnL`.

        This method should calculate the value of the likelihood for
        the model set `mset`.

        :param _: unused, but required by interface
        """
        loglike = self.likelihood.compute_loglike(self.tools)
        return -2.0 * loglike


class NumCosmoGaussCov(Ncm.DataGaussCov):
    """NumCosmoGaussCov is a subclass of Ncm.DataGaussCov.

    This subclass implements NumCosmo likelihood object virtual methods using the
    prefix `do_`. This class implements a Gaussian likelihood.
    """

    __gtype_name__ = "FirecrownNumCosmoGaussCov"

    def __init__(self) -> None:
        """Initialize a NumCosmoGaussCov object.

        This class is a subclass of Ncm.DataGaussCov and implements NumCosmo
        likelihood object virtual methods using the prefix `do_`. This class
        implements a Gaussian likelihood.

        Due to the way GObject works, the constructor must have a `**kwargs`
        argument, and the properties must be set after construction.

        In python one should use the `new_from_likelihood` method to create a
        NumCosmoGaussCov object from a ConstGaussian object. This constructor
        has the correct signature for type checking.
        """
        super().__init__()
        self.likelihood: ConstGaussian
        self.tools: ModelingTools
        self.dof: int
        self.len: int
        self._model_list: list[str]
        self._nc_mapping: MappingNumCosmo
        self._likelihood_source: None | str = None
        self._likelihood_build_parameters: None | NamedParameters = None
        self._starting_deserialization: bool = False

    def _get_model_list(self) -> list[str]:
        """Return the list of models.

        :return: the current names of the models
        """
        return self._model_list

    def _set_model_list(self, value: list[str]) -> None:
        """Set the list of models.

        :param value: the new list of model names
        """
        self._model_list = value

    model_list = GObject.Property(
        type=GObject.TYPE_STRV,  # type: ignore
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_model_list,
        setter=_set_model_list,
    )

    def _get_nc_mapping(self) -> MappingNumCosmo:
        """Return the :class:`MappingNumCosmo` object.

        :return: the current value of the mapping
        """
        return self._nc_mapping

    def _set_nc_mapping(self, value: MappingNumCosmo):
        """Set the MappingNumCosmo object.

        :param: the new value for the MappingNumCosmo object
        """
        self._nc_mapping = value

    nc_mapping = GObject.Property(
        type=MappingNumCosmo,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_nc_mapping,
        setter=_set_nc_mapping,
    )

    def _configure_object(self) -> None:
        """Configure the object."""
        assert self.likelihood is not None

        cov = self.likelihood.get_cov()
        nrows, ncols = cov.shape
        assert nrows == ncols

        self.set_size(nrows)
        self.dof = nrows
        self.len = nrows
        self.peek_cov().set_from_array(  # pylint: disable-msg=no-member
            cov.flatten().tolist()
        )

        data_vector = self.likelihood.get_data_vector()
        assert len(data_vector) == ncols
        self.peek_mean().set_array(  # pylint: disable-msg=no-member
            data_vector.ravel().tolist()
        )

        self.set_init(True)

    def _set_likelihood_from_factory(self) -> None:
        """Deserialize the likelihood."""
        assert self._likelihood_source is not None
        assert self._likelihood_build_parameters is not None
        likelihood, tools = load_likelihood(
            self._likelihood_source, self._likelihood_build_parameters
        )
        assert isinstance(likelihood, ConstGaussian)
        assert isinstance(tools, ModelingTools)
        self.likelihood = likelihood
        self.tools = tools
        self._configure_object()

    def _get_likelihood_source(self) -> None | str:
        """Return the likelihood string defining the factory function.

        :return: the filename of the likelihood factory function
        """
        return self._likelihood_source

    def _set_likelihood_source(self, value: None | str) -> None:
        """Set the likelihood string defining the factory function.

        :param value: the filename of the likelihood factory function
        """
        if value is not None:
            self._likelihood_source = value
            if self._starting_deserialization:
                self._set_likelihood_from_factory()
                self._starting_deserialization = False
            else:
                self._starting_deserialization = True

    likelihood_source = GObject.Property(
        type=str,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_likelihood_source,
        setter=_set_likelihood_source,
    )

    def _get_likelihood_build_parameters(self) -> None | Ncm.VarDict:
        """Return the likelihood build parameters.

        :return: the likelihood build parameters
        """
        if self._likelihood_build_parameters is None:
            return None
        return dict_to_var_dict(
            self._likelihood_build_parameters.convert_to_basic_dict()
        )

    def _set_likelihood_build_parameters(self, value: None | Ncm.VarDict) -> None:
        """Set the likelihood build parameters.

        :param value: the likelihood build parameters
        """
        self._likelihood_build_parameters = NamedParameters()
        if value is not None:
            self._likelihood_build_parameters.set_from_basic_dict(
                var_dict_to_dict(value)
            )
        if self._starting_deserialization:
            self._set_likelihood_from_factory()
            self._starting_deserialization = False
        else:
            self._starting_deserialization = True

    likelihood_build_parameters = GObject.Property(
        type=Ncm.VarDict,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_likelihood_build_parameters,
        setter=_set_likelihood_build_parameters,
    )

    @classmethod
    def new_from_likelihood(
        cls,
        likelihood: ConstGaussian,
        model_list: list[str],
        tools: ModelingTools,
        nc_mapping: MappingNumCosmo,
        likelihood_source: None | str = None,
        likelihood_build_parameters: None | NamedParameters = None,
    ):
        """Initialize a NumCosmoGaussCov object.

        This object represents a Gaussian likelihood with a constant covariance.
        :param likelihood: the likelihood object
        :param model_list: the list of model names
        :param nc_mapping: the mapping object
        :param likelihood_source: the filename for the likelihood factory function
        :param likelihood_build_parameters: the build parameters of the likelihood
        """
        cov = likelihood.get_cov()
        nrows, ncols = cov.shape
        assert nrows == ncols

        nc_gauss_cov: NumCosmoGaussCov = GObject.new(
            cls,
            model_list=model_list,
            nc_mapping=nc_mapping,
            likelihood_source=None,
            likelihood_build_parameters=None,
        )

        assert isinstance(nc_gauss_cov, NumCosmoGaussCov)

        nc_gauss_cov.likelihood = likelihood
        nc_gauss_cov.tools = tools
        # pylint: disable=protected-access
        nc_gauss_cov._likelihood_source = likelihood_source
        nc_gauss_cov._likelihood_build_parameters = likelihood_build_parameters
        nc_gauss_cov._configure_object()
        # pylint: enable=protected-access

        return nc_gauss_cov

    def do_get_length(self) -> int:  # pylint: disable-msg=arguments-differ
        """Implements the virtual `Ncm.Data` method `get_length`.

        :return: the number of data points in the likelihood
        """
        return self.len

    def do_get_dof(self) -> int:  # pylint: disable-msg=arguments-differ
        """Implements the virtual `Ncm.Data` method `get_dof`.

        :return: the number of degrees of freedom in the likelihood
        """
        return self.dof

    def do_begin(self) -> None:  # pylint: disable-msg=arguments-differ
        """Implements the virtual `Ncm.Data` method `begin`.

        This method usually do some groundwork in the data
        before the actual calculations. For example, if the likelihood
        involves the decomposition of a constant matrix, it can be done
        during `begin` once and then used afterwards.
        """

    def do_prepare(  # pylint: disable-msg=arguments-differ
        self, mset: Ncm.MSet
    ) -> None:
        """Implements the virtual method Ncm.Data `prepare`.

        This method should do all the necessary calculations using mset
        to be able to calculate the likelihood afterwards.
        :param mset: the model set
        """
        self.dof = self.len - mset.fparams_len()
        self.likelihood.reset()
        self.tools.reset()

        self._nc_mapping.set_params_from_numcosmo(mset, self.tools.ccl_factory)
        params_map = self._nc_mapping.create_params_map(self._model_list, mset)

        self.likelihood.update(params_map)
        self.tools.update(params_map)
        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            self.tools.prepare(
                calculator_args=self._nc_mapping.calculate_ccl_args(mset)
            )
        else:
            self.tools.prepare()

    # pylint: disable-next=arguments-differ
    def do_mean_func(self, _, vp) -> None:
        """Implements the virtual `Ncm.DataGaussCov` method `mean_func`.

        This method should compute the theoretical mean for the gaussian
        distribution.

        :param _: unused, but required by interface
        :param vp: the vector to set
        """
        theory_vector = self.likelihood.compute_theory_vector(self.tools)
        vp.set_array(theory_vector)


# These commands creates GObject types for the defined classes, enabling their use
# within the NumCosmo framework. It is essential to call these functions before
# initializing NumCosmo with the Ncm.init_cfg() function, as failure to do so
# will cause issues with MPI jobs using these objects.
GObject.type_register(MappingNumCosmo)
GObject.type_register(NumCosmoData)
GObject.type_register(NumCosmoGaussCov)


class NumCosmoFactory:
    """NumCosmo likelihood class.

    This class provide the necessary factory methods
    to create NumCosmo+firecrown likelihoods.
    """

    def __init__(
        self,
        likelihood_source: str,
        build_parameters: NamedParameters,
        mapping: MappingNumCosmo,
        model_list: list[str],
    ) -> None:
        """Initialize a NumCosmoFactory.

        :param likelihood_source: the filename for the likelihood factory function
        :param build_parameters: the build parameters
        :param mapping: the mapping
        :param model_list: the model list
        """
        likelihood, tools = load_likelihood(likelihood_source, build_parameters)

        self.data: NumCosmoGaussCov | NumCosmoData
        self.mapping: MappingNumCosmo = mapping
        if isinstance(likelihood, ConstGaussian):
            self.data = NumCosmoGaussCov.new_from_likelihood(
                likelihood,
                model_list,
                tools,
                mapping,
                likelihood_source,
                build_parameters,
            )
        else:
            self.data = NumCosmoData.new_from_likelihood(
                likelihood,
                model_list,
                tools,
                mapping,
                likelihood_source,
                build_parameters,
            )

    def get_data(self) -> NumCosmoGaussCov | NumCosmoData:
        """This method return the appropriate Ncm.Data class to be used by NumCosmo.

        :return: the data used by NumCosmo
        """
        return self.data

    def get_mapping(self) -> MappingNumCosmo:
        """This method return the current MappingNumCosmo.

        :return: the current mapping.
        """
        return self.mapping

    def get_firecrown_likelihood(self) -> Likelihood:
        """This method returns the Firecrown Likelihood.

        :return: the likelihood
        """
        return self.data.likelihood
