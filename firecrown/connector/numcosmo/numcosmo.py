"""Support of Firecrown likelihood use in NumCosmo.

The subpackages and modules in this package depend upon NumCosmo, and can not
be used without an installation of NumCosmo.

"""

from typing import Dict, Union, List, Any, Optional
import pickle
import base64
import numpy as np
import pyccl as ccl

from numcosmo_py import Nc, Ncm, GObject

from firecrown.likelihood.likelihood import load_likelihood
from firecrown.likelihood.likelihood import Likelihood
from firecrown.likelihood.likelihood import NamedParameters
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian
from firecrown.parameters import ParamsMap
from firecrown.connector.mapping import Mapping, build_ccl_background_dict
from firecrown.modeling_tools import ModelingTools


class MappingNumCosmo(GObject.Object):
    """Mapping support for NumCosmo, this is a subclass of Mapping that
    provides a mapping from a NumCosmo Cosmological model to a CCL cosmology.
    It alsos convert NumCosmo models to `ParamsMap`s."""

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

    @GObject.Property(
        type=str,
        default="default",
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT_ONLY,
    )
    def mapping_name(self) -> Mapping:
        """Return the Mapping object."""
        return self._mapping_name

    @mapping_name.setter  # type: ignore
    def mapping_name(self, value: str):
        """Set the Mapping object."""
        self._mapping_name = value
        self.mapping = Mapping()

    @GObject.Property(type=bool, default=False, flags=GObject.ParamFlags.READWRITE)
    def require_nonlinear_pk(self) -> bool:
        """Return whether nonlinear power spectra are required."""
        return self.mapping.require_nonlinear_pk

    @require_nonlinear_pk.setter  # type: ignore
    def require_nonlinear_pk(self, value: bool):
        """Set whether nonlinear power spectra are required."""
        self.mapping.require_nonlinear_pk = value

    @GObject.Property(type=Nc.PowspecML, flags=GObject.ParamFlags.READWRITE)
    def p_ml(self) -> Optional[Nc.PowspecML]:
        """Return the NumCosmo PowspecML object."""
        return self._p_ml

    @p_ml.setter  # type: ignore
    def p_ml(self, value: Optional[Nc.PowspecML]):
        """Set the NumCosmo PowspecML object."""
        self._p_ml: Optional[Nc.PowspecML] = value

    @GObject.Property(type=Nc.PowspecMNL, flags=GObject.ParamFlags.READWRITE)
    def p_mnl(self) -> Optional[Nc.PowspecMNL]:
        """Return the NumCosmo PowspecMNL object."""
        return self._p_mnl

    @p_mnl.setter  # type: ignore
    def p_mnl(self, value: Optional[Nc.PowspecMNL]):
        """Set the NumCosmo PowspecMNL object."""
        self._p_mnl: Optional[Nc.PowspecMNL] = value

    @GObject.Property(type=Nc.Distance, flags=GObject.ParamFlags.READWRITE)
    def dist(self) -> Nc.Distance:
        """Return the NumCosmo Distance object."""
        return self._dist

    @dist.setter  # type: ignore
    def dist(self, value: Nc.Distance):
        """Set the NumCosmo Distance object."""
        self._dist: Nc.Distance = value

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

        m_nu: Union[float, List[float]] = 0.0
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
        ccl_args: Dict[str, Any] = {}
        hi_cosmo = mset.peek(Nc.HICosmo.id())
        assert isinstance(hi_cosmo, Nc.HICosmo)

        if self._p_ml:
            p_m_spline = self._p_ml.get_spline_2d(hi_cosmo)
            z = np.array(p_m_spline.xv.dup_array())
            k = np.array(p_m_spline.yv.dup_array())

            scale = self.mapping.redshift_to_scale_factor(z)
            p_k = np.transpose(
                np.array(p_m_spline.zm.dup_array()).reshape(len(k), len(z))
            )
            p_k = self.mapping.redshift_to_scale_factor_p_k(p_k)

            ccl_args["pk_linear"] = {
                "a": scale,
                "k": k,
                "delta_matter:delta_matter": p_k,
            }

        if self._p_mnl:
            p_mnl_spline = self._p_mnl.get_spline_2d(hi_cosmo)
            z = np.array(p_mnl_spline.xv.dup_array())
            k = np.array(p_mnl_spline.yv.dup_array())

            scale_mpnl = self.mapping.redshift_to_scale_factor(z)
            p_mnl = np.transpose(
                np.array(p_mnl_spline.zm.dup_array()).reshape(len(k), len(z))
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

    def create_params_map(self, model_list: List[str], mset: Ncm.MSet) -> ParamsMap:
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
    object virtual methods using the prefix :python:`do_`. This class implement
    a general likelihood."""

    __gtype_name__ = "FirecrownNumCosmoData"

    def __init__(
        self,
        likelihood: Likelihood,
        model_list: List[str],
        tools: ModelingTools,
        nc_mapping: MappingNumCosmo,
    ):
        super().__init__()
        self.likelihood: Likelihood = likelihood
        self.model_list: List[str] = model_list
        self.tools: ModelingTools = tools
        self.dof: int = 100
        self.len: int = 100
        self.nc_mapping = nc_mapping
        self.ccl_cosmo: Optional[ccl.Cosmology] = None
        self.set_init(True)

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

        self.nc_mapping.set_params_from_numcosmo(mset)
        ccl_args = self.nc_mapping.calculate_ccl_args(mset)
        self.ccl_cosmo = ccl.CosmologyCalculator(
            **self.nc_mapping.mapping.asdict(), **ccl_args
        )
        params_map = self.nc_mapping.create_params_map(self.model_list, mset)

        self.likelihood.update(params_map)
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
    object virtual methods using the prefix :python:`do_`. This class implement
    a Gaussian likelihood."""

    __gtype_name__ = "FirecrownNumCosmoGaussCov"

    def __init__(self):
        """Initialize a NumCosmoGaussCov object. This class is a subclass of
        Ncm.DataGaussCov and implements NumCosmo likelihood object virtual
        methods using the prefix :python:`do_`. This class implement a Gaussian
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
        self._model_list: List[str]
        self._nc_mapping: MappingNumCosmo
        self._likelihood_str: Optional[str] = None

    @GObject.Property(
        type=GObject.TYPE_STRV,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
    )
    def model_list(self) -> List[str]:
        """Return the list of models."""
        return self._model_list

    @model_list.setter  # type: ignore
    def model_list(self, value: List[str]):
        """Set the list of models."""
        self._model_list = value

    @GObject.Property(
        type=MappingNumCosmo,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
    )
    def nc_mapping(self) -> MappingNumCosmo:
        """Return the MappingNumCosmo object."""
        return self._nc_mapping

    @nc_mapping.setter  # type: ignore
    def nc_mapping(self, value: MappingNumCosmo):
        """Set the MappingNumCosmo object."""
        self._nc_mapping = value

    @GObject.Property(
        type=str,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
    )
    def likelihood_str(self) -> Optional[str]:
        """Return the likelihood string."""
        return self._likelihood_str

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

    @likelihood_str.setter  # type: ignore
    def likelihood_str(self, value: Optional[str]):
        """Set the likelihood string."""
        self._likelihood_str = value
        if value is not None:
            likelihood_source, build_parameters = pickle.loads(
                base64.b64decode(value.encode("ascii"))
            )
            likelihood, tools = load_likelihood(likelihood_source, build_parameters)
            assert isinstance(likelihood, ConstGaussian)
            self.likelihood = likelihood
            self.tools = tools
            self._configure_object()

    @classmethod
    def new_from_likelihood(
        cls,
        likelihood: ConstGaussian,
        model_list: List[str],
        tools: ModelingTools,
        nc_mapping: MappingNumCosmo,
        likelihood_str: Optional[str] = None,
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
            likelihood_str=None,
        )

        nc_gauss_cov.likelihood = likelihood
        nc_gauss_cov.tools = tools
        # pylint: disable=protected-access
        nc_gauss_cov._likelihood_str = likelihood_str
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
# within the NumCosmo framework. It is essential to call this module before
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
        model_list: List[str],
    ):
        likelihood, tools = load_likelihood(likelihood_source, build_parameters)

        likelihood_str = base64.b64encode(
            pickle.dumps((likelihood_source, build_parameters))
        ).decode("ascii")

        self.mapping: MappingNumCosmo = mapping
        if isinstance(likelihood, ConstGaussian):
            self.data: Ncm.Data = NumCosmoGaussCov.new_from_likelihood(
                likelihood, model_list, tools, mapping, likelihood_str
            )
        else:
            self.data = NumCosmoData(likelihood, model_list, tools, mapping)

    def get_data(self) -> Ncm.Data:
        """This method return the appropriated Ncm.Data class to be used by NumCosmo."""
        return self.data

    def get_mapping(self) -> MappingNumCosmo:
        """This method return the current MappingNumCosmo."""
        return self.mapping
