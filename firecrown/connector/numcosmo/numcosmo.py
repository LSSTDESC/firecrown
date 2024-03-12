"""Support of Firecrown likelihood use in NumCosmo.

The subpackages and modules in this package depend upon NumCosmo, and can not
be used without an installation of NumCosmo.

"""

from typing import Union, Any, Optional
import numpy as np
import pyccl as ccl

from numcosmo_py import Nc, Ncm, GObject, var_dict_to_dict, dict_to_var_dict

from firecrown.likelihood.likelihood import load_likelihood
from firecrown.likelihood.likelihood import Likelihood
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from firecrown.connector.mapping import Mapping, build_ccl_background_dict
from firecrown.modeling_tools import ModelingTools


class MappingNumCosmo(GObject.Object):
    """Mapping support for NumCosmo, this is a subclass of :class:`Mapping` that
    provides a mapping from a NumCosmo Cosmological model to a CCL cosmology.
    It also converts NumCosmo models to :class:`ParamsMap` objects."""

    __gtype_name__ = "FirecrownMappingNumCosmo"

    def __init__(
        self,
        require_nonlinear_pk: bool = False,
        p_ml: Optional[Nc.PowspecML] = None,
        p_mnl: Optional[Nc.PowspecMNL] = None,
        dist: Optional[Nc.Distance] = None,
    ):
        """Initialize a MappingNumCosmo object."""
        super().__init__(  # type: ignore
            require_nonlinear_pk=require_nonlinear_pk, p_ml=p_ml, p_mnl=p_mnl, dist=dist
        )
        self.mapping: Mapping
        self._mapping_name: str
        self._p_ml: Optional[Nc.PowspecML]
        self._p_mnl: Optional[Nc.PowspecMNL]
        self._dist: Nc.Distance

        if not hasattr(self, "_p_ml"):
            self._p_ml = None

        if not hasattr(self, "_p_mnl"):
            self._p_mnl = None

    def _get_mapping_name(self) -> str:
        """Return the mapping name."""
        return self._mapping_name

    def _set_mapping_name(self, value: str):
        """Set the mapping name."""
        self._mapping_name = value
        self.mapping = Mapping()

    mapping_name = GObject.Property(
        type=str,
        default="default",
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT_ONLY,
        getter=_get_mapping_name,
        setter=_set_mapping_name,
    )

    def _get_require_nonlinear_pk(self) -> bool:
        """Return whether nonlinear power spectra are required."""
        return self.mapping.require_nonlinear_pk

    def _set_require_nonlinear_pk(self, value: bool):
        """Set whether nonlinear power spectra are required."""
        self.mapping.require_nonlinear_pk = value

    require_nonlinear_pk = GObject.Property(
        type=bool,
        default=False,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_require_nonlinear_pk,
        setter=_set_require_nonlinear_pk,
    )

    def _get_p_ml(self) -> Optional[Nc.PowspecML]:
        """Return the NumCosmo PowspecML object."""
        return self._p_ml

    def _set_p_ml(self, value: Optional[Nc.PowspecML]):
        """Set the NumCosmo PowspecML object."""
        self._p_ml = value

    p_ml = GObject.Property(
        type=Nc.PowspecML,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_p_ml,
        setter=_set_p_ml,
    )

    def _get_p_mnl(self) -> Optional[Nc.PowspecMNL]:
        """Return the NumCosmo PowspecMNL object."""
        return self._p_mnl

    def _set_p_mnl(self, value: Optional[Nc.PowspecMNL]):
        """Set the NumCosmo PowspecMNL object."""
        self._p_mnl = value

    p_mnl = GObject.Property(
        type=Nc.PowspecMNL,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_p_mnl,
        setter=_set_p_mnl,
    )

    def _get_dist(self) -> Optional[Nc.Distance]:
        """Return the NumCosmo Distance object."""
        return self._dist

    def _set_dist(self, value: Nc.Distance):
        """Set the NumCosmo Distance object."""
        self._dist = value

    dist = GObject.Property(
        type=Nc.Distance,
        flags=GObject.ParamFlags.READWRITE,
        getter=_get_dist,
        setter=_set_dist,
    )

    def set_params_from_numcosmo(
        self, mset: Ncm.MSet
    ):  # pylint: disable-msg=too-many-locals
        """Return a PyCCLCosmologyConstants object with parameters equivalent to
        those read from NumCosmo."""

        hi_cosmo = mset.peek(Nc.HICosmo.id())
        assert isinstance(hi_cosmo, Nc.HICosmo)

        if self._p_ml is not None:
            self._p_ml.prepare_if_needed(hi_cosmo)
        if self._p_mnl is not None:
            self._p_mnl.prepare_if_needed(hi_cosmo)
        self._dist.prepare_if_needed(hi_cosmo)

        h = hi_cosmo.h()
        Omega_b = hi_cosmo.Omega_b0()
        Omega_c = hi_cosmo.Omega_c0()
        Omega_k = hi_cosmo.Omega_k0()
        Neff = hi_cosmo.Neff()
        T_gamma0 = hi_cosmo.T_gamma0()

        m_nu: Union[float, list[float]] = 0.0
        if hi_cosmo.NMassNu() == 0:
            m_nu_type = "normal"
        else:
            m_nu_type = "list"
            m_nu = [0.0, 0.0, 0.0]
            assert hi_cosmo.NMassNu() <= 3
            for i in range(hi_cosmo.NMassNu()):
                nu_info = hi_cosmo.MassNuInfo(i)
                m_nu[i] = nu_info[0]

        if isinstance(hi_cosmo, Nc.HICosmoDEXcdm):
            w0 = hi_cosmo.props.w
            wa = 0.0
        elif isinstance(hi_cosmo, Nc.HICosmoDECpl):
            w0 = hi_cosmo.props.w0
            wa = hi_cosmo.props.w1
        else:
            raise ValueError(f"NumCosmo object {type(hi_cosmo)} not supported.")

        hiprim = hi_cosmo.peek_submodel_by_mid(Nc.HIPrim.id())
        if not hiprim:
            raise ValueError("NumCosmo object must include a HIPrim object.")
        if not isinstance(hiprim, Nc.HIPrimPowerLaw):
            raise ValueError(
                f"NumCosmo HIPrim object type {type(hiprim)} not supported."
            )

        A_s = hiprim.SA_Ampl()
        n_s = hiprim.props.n_SA

        # pylint: disable=duplicate-code
        self.mapping.set_params(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            A_s=A_s,
            n_s=n_s,
            Omega_k=Omega_k,
            Neff=Neff,
            m_nu=m_nu,
            m_nu_type=m_nu_type,
            w0=w0,
            wa=wa,
            T_CMB=T_gamma0,
        )
        # pylint: enable=duplicate-code

    def calculate_ccl_args(self, mset: Ncm.MSet):  # pylint: disable-msg=too-many-locals
        """Calculate the arguments necessary for CCL for this sample."""
        ccl_args: dict[str, Any] = {}
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

            ccl_args["pk_linear"] = {
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

            ccl_args["pk_nonlin"] = {
                "a": scale_mpnl,
                "k": k,
                "delta_matter:delta_matter": p_mnl,
            }
        elif self.mapping.require_nonlinear_pk:
            ccl_args["nonlinear_model"] = "halofit"
        else:
            ccl_args["nonlinear_model"] = None

        d_spline = self._dist.comoving_distance_spline.peek_spline()
        z_dist = np.array(d_spline.get_xv().dup_array())
        c_dist = np.array(d_spline.get_yv().dup_array())

        chi = np.flip(c_dist) * hi_cosmo.RH_Mpc()
        scale_distances = self.mapping.redshift_to_scale_factor(z_dist)
        h_over_h0 = np.array([hi_cosmo.E(z) for z in reversed(z_dist)])

        # Too many points in the redshift spline can result in scale factors
        # that are too close together for CCL to handle. This checks for
        # duplicate scale factors and removes them.
        a_unique, a_unique_indices = np.unique(scale_distances, return_index=True)
        scale_distances = a_unique
        chi = chi[a_unique_indices]
        h_over_h0 = h_over_h0[a_unique_indices]

        ccl_args["background"] = build_ccl_background_dict(
            a=scale_distances, chi=chi, h_over_h0=h_over_h0
        )

        ccl_args["background"] = {
            "a": scale_distances,
            "chi": chi,
            "h_over_h0": h_over_h0,
        }
        return ccl_args

    def create_params_map(self, model_list: list[str], mset: Ncm.MSet) -> ParamsMap:
        """Create a ParamsMap from a NumCosmo MSet."""

        params_map = ParamsMap()
        for model_ns in model_list:
            mid = mset.get_id_by_ns(model_ns)
            if mid < 0:
                raise RuntimeError(
                    f"Model name {model_ns} was not found in the model set."
                )
            model = mset.peek(mid)
            if model is None:
                raise RuntimeError(f"Model {model_ns} was not found in the model set.")
            param_names = model.param_names()
            model_dict = {
                param: model.param_get_by_name(param) for param in param_names
            }
            shared_keys = set(model_dict).intersection(params_map)
            if len(shared_keys) > 0:
                raise RuntimeError(
                    f"The following keys `{shared_keys}` appear "
                    f"in more than one model used by the "
                    f"module {model_list}."
                )
            params_map = ParamsMap({**params_map, **model_dict})

        return params_map


class NumCosmoData(Ncm.Data):
    """NumCosmoData is a subclass of Ncm.Data and implements NumCosmo likelihood
    object virtual methods using the prefix `do_`. This class implements
    a general likelihood."""

    __gtype_name__ = "FirecrownNumCosmoData"

    def __init__(self):
        super().__init__()
        self.likelihood: Likelihood
        self.tools: ModelingTools
        self.ccl_cosmo: Optional[ccl.Cosmology] = None
        self._model_list: list[str]
        self._nc_mapping: MappingNumCosmo
        self._likelihood_source: Optional[str] = None
        self._likelihood_build_parameters: Optional[NamedParameters] = None
        self._starting_deserialization: bool = False
        self.dof: int = 100
        self.len: int = 100
        self.set_init(True)

    def _get_model_list(self) -> list[str]:
        """Return the list of models."""
        return self._model_list

    def _set_model_list(self, value: list[str]):
        """Set the list of models."""
        self._model_list = value

    model_list = GObject.Property(
        type=GObject.TYPE_STRV,  # type: ignore
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_model_list,
        setter=_set_model_list,
    )

    def _get_nc_mapping(self) -> MappingNumCosmo:
        """Return the MappingNumCosmo object."""
        return self._nc_mapping

    def _set_nc_mapping(self, value: MappingNumCosmo):
        """Set the MappingNumCosmo object."""
        self._nc_mapping = value

    nc_mapping = GObject.Property(
        type=MappingNumCosmo,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_nc_mapping,
        setter=_set_nc_mapping,
    )

    def _set_likelihood_from_factory(self):
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

    def _get_likelihood_source(self) -> Optional[str]:
        """Return the likelihood string defining the factory function."""
        return self._likelihood_source

    def _set_likelihood_source(self, value: Optional[str]):
        """Set the likelihood string defining the factory function."""

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

    def _get_likelihood_build_parameters(self) -> Optional[Ncm.VarDict]:
        """Return the likelihood build parameters."""
        if self._likelihood_build_parameters is None:
            return None
        return dict_to_var_dict(
            self._likelihood_build_parameters.convert_to_basic_dict()
        )

    def _set_likelihood_build_parameters(self, value: Optional[Ncm.VarDict]):
        """Set the likelihood build parameters."""
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
        likelihood_source: Optional[str] = None,
        likelihood_build_parameters: Optional[NamedParameters] = None,
    ):
        """Initialize a NumCosmoGaussCov object representing a Gaussian likelihood
        with a constant covariance."""

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

    def do_get_length(self):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual Ncm.Data method get_length.
        """
        return self.len

    def do_get_dof(self):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual Ncm.Data method get_dof.
        """
        return self.dof

    def do_begin(self):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual Ncm.Data method `begin`.
        This method usually do some groundwork in the data
        before the actual calculations. For example, if the likelihood
        involves the decomposition of a constant matrix, it can be done
        during `begin` once and then used afterwards.
        """

    def do_prepare(self, mset: Ncm.MSet):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual method Ncm.Data `prepare`.
        This method should do all the necessary calculations using mset
        to be able to calculate the likelihood afterwards.
        """
        self.dof = self.len - mset.fparams_len()
        self.likelihood.reset()
        self.tools.reset()

        self._nc_mapping.set_params_from_numcosmo(mset)
        ccl_args = self._nc_mapping.calculate_ccl_args(mset)
        self.ccl_cosmo = ccl.CosmologyCalculator(
            **self._nc_mapping.mapping.asdict(), **ccl_args
        )
        params_map = self._nc_mapping.create_params_map(self.model_list, mset)

        self.likelihood.update(params_map)
        self.tools.update(params_map)
        self.tools.prepare(self.ccl_cosmo)

    def do_m2lnL_val(self, _):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual method `m2lnL`.
        This method should calculate the value of the likelihood for
        the model set `mset`.
        """
        loglike = self.likelihood.compute_loglike(self.tools)
        return -2.0 * loglike


class NumCosmoGaussCov(Ncm.DataGaussCov):
    """NumCosmoData is a subclass of Ncm.Data and implements NumCosmo likelihood
    object virtual methods using the prefix `do_`. This class implements
    a Gaussian likelihood."""

    __gtype_name__ = "FirecrownNumCosmoGaussCov"

    def __init__(self):
        """Initialize a NumCosmoGaussCov object. This class is a subclass of
        Ncm.DataGaussCov and implements NumCosmo likelihood object virtual
        methods using the prefix `do_`. This class implements a Gaussian
        likelihood.

        Due to the way GObject works, the constructor must have a `**kwargs`
        argument, and the properties must be set after construction.

        In python one should use the `new_from_likelihood` method to create a
        NumCosmoGaussCov object from a ConstGaussian object. This constuctor
        has the correct signature for type checking.
        """
        super().__init__()
        self.likelihood: ConstGaussian
        self.tools: ModelingTools
        self.ccl_cosmo: ccl.Cosmology
        self.dof: int
        self.len: int
        self._model_list: list[str]
        self._nc_mapping: MappingNumCosmo
        self._likelihood_source: Optional[str] = None
        self._likelihood_build_parameters: Optional[NamedParameters] = None
        self._starting_deserialization: bool = False

    def _get_model_list(self) -> list[str]:
        """Return the list of models."""
        return self._model_list

    def _set_model_list(self, value: list[str]):
        """Set the list of models."""
        self._model_list = value

    model_list = GObject.Property(
        type=GObject.TYPE_STRV,  # type: ignore
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_model_list,
        setter=_set_model_list,
    )

    def _get_nc_mapping(self) -> MappingNumCosmo:
        """Return the :class:`MappingNumCosmo` object."""
        return self._nc_mapping

    def _set_nc_mapping(self, value: MappingNumCosmo):
        """Set the MappingNumCosmo object."""
        self._nc_mapping = value

    nc_mapping = GObject.Property(
        type=MappingNumCosmo,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_nc_mapping,
        setter=_set_nc_mapping,
    )

    def _configure_object(self):
        """Configure the object."""
        assert self.likelihood is not None

        cov = self.likelihood.get_cov()
        nrows, ncols = cov.shape
        assert nrows == ncols

        self.set_size(nrows)
        self.ccl_cosmo = None
        self.dof = nrows
        self.len = nrows
        self.peek_cov().set_from_array(
            cov.flatten().tolist()
        )  # pylint: disable-msg=no-member

        data_vector = self.likelihood.get_data_vector()
        assert len(data_vector) == ncols
        self.peek_mean().set_array(
            data_vector.tolist()
        )  # pylint: disable-msg=no-member

        self.set_init(True)

    def _set_likelihood_from_factory(self):
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

    def _get_likelihood_source(self) -> Optional[str]:
        """Return the likelihood string defining the factory function."""
        return self._likelihood_source

    def _set_likelihood_source(self, value: Optional[str]):
        """Set the likelihood string defining the factory function."""

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

    def _get_likelihood_build_parameters(self) -> Optional[Ncm.VarDict]:
        """Return the likelihood build parameters."""
        if self._likelihood_build_parameters is None:
            return None
        return dict_to_var_dict(
            self._likelihood_build_parameters.convert_to_basic_dict()
        )

    def _set_likelihood_build_parameters(self, value: Optional[Ncm.VarDict]):
        """Set the likelihood build parameters."""
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
        likelihood_source: Optional[str] = None,
        likelihood_build_parameters: Optional[NamedParameters] = None,
    ):
        """Initialize a NumCosmoGaussCov object representing a Gaussian likelihood
        with a constant covariance."""

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

    def do_get_length(self):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual `Ncm.Data` method `get_length`.
        """
        return self.len

    def do_get_dof(self):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual `Ncm.Data` method `get_dof`.
        """
        return self.dof

    def do_begin(self):  # pylint: disable-msg=arguments-differ
        """
        # Implements the virtual `Ncm.Data` method `begin`.
        # This method usually do some groundwork in the data
        # before the actual calculations. For example, if the likelihood
        # involves the decomposition of a constant matrix, it can be done
        # during `begin` once and then used afterwards.
        """

    def do_prepare(self, mset: Ncm.MSet):  # pylint: disable-msg=arguments-differ
        """
        Implements the virtual method Ncm.Data `prepare`.
        This method should do all the necessary calculations using mset
        to be able to calculate the likelihood afterwards.
        """
        self.dof = self.len - mset.fparams_len()
        self.likelihood.reset()
        self.tools.reset()

        self._nc_mapping.set_params_from_numcosmo(mset)
        ccl_args = self._nc_mapping.calculate_ccl_args(mset)

        self.ccl_cosmo = ccl.CosmologyCalculator(
            **self._nc_mapping.mapping.asdict(), **ccl_args
        )
        params_map = self._nc_mapping.create_params_map(self._model_list, mset)

        self.likelihood.update(params_map)
        self.tools.update(params_map)
        self.tools.prepare(self.ccl_cosmo)

    # pylint: disable-next=arguments-differ
    def do_mean_func(self, _, mean_vector):
        """
        Implements the virtual `Ncm.DataGaussCov` method `mean_func`.
        This method should compute the theoretical mean for the gaussian
        distribution.
        """
        theory_vector = self.likelihood.compute_theory_vector(self.tools)
        mean_vector.set_array(theory_vector)


# These commands creates GObject types for the defined classes, enabling their use
# within the NumCosmo framework. It is essential to call these functions before
# initializing NumCosmo with the Ncm.init_cfg() function, as failure to do so
# will cause issues with MPI jobs using these objects.
GObject.type_register(MappingNumCosmo)
GObject.type_register(NumCosmoData)
GObject.type_register(NumCosmoGaussCov)


class NumCosmoFactory:
    """NumCosmo likelihood class. This class provide the necessary factory methods
    to create NumCosmo+firecrown likelihoods."""

    def __init__(
        self,
        likelihood_source: str,
        build_parameters: NamedParameters,
        mapping: MappingNumCosmo,
        model_list: list[str],
    ):
        likelihood, tools = load_likelihood(likelihood_source, build_parameters)

        self.data: Union[NumCosmoGaussCov, NumCosmoData]
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

    def get_data(self) -> Ncm.Data:
        """This method return the appropriated Ncm.Data class to be used by NumCosmo."""
        return self.data

    def get_mapping(self) -> MappingNumCosmo:
        """This method return the current MappingNumCosmo."""
        return self.mapping

    def get_firecrown_likelihood(self) -> Likelihood:
        """This method returns the firecrown Likelihood."""
        return self.data.likelihood
