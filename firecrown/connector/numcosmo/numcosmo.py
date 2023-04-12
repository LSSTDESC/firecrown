"""Support of Firecrown likelihood use in NumCosmo.

The subpackages and modules in this package depend upon NumCosmo, and can not
be used without an installation of NumCosmo.

"""

from typing import Dict, Union, List, Any, Optional
import numpy as np
import pyccl as ccl

import gi

gi.require_version("NumCosmo", "1.0")
gi.require_version("NumCosmoMath", "1.0")

# pylint: disable=wrong-import-position
from gi.repository import NumCosmo as Nc  # noqa: E402
from gi.repository import NumCosmoMath as Ncm  # noqa: E402

from firecrown.likelihood.likelihood import load_likelihood  # noqa: E402
from firecrown.likelihood.likelihood import Likelihood  # noqa: E402
from firecrown.likelihood.likelihood import NamedParameters  # noqa: E402
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian  # noqa: E402
from firecrown.parameters import ParamsMap  # noqa: E402
from firecrown.connector.mapping import Mapping, build_ccl_background_dict  # noqa: E402
from firecrown.modeling_tools import ModelingTools  # noqa: E402

# pylint: enable=wrong-import-position


class MappingNumCosmo(Mapping):
    """Mapping support for NumCosmo."""

    def __init__(
        self,
        *,
        require_nonlinear_pk: bool = False,
        p_ml: Optional[Nc.PowspecML] = None,
        p_mnl: Optional[Nc.PowspecMNL] = None,
        dist: Nc.Distance,
    ):
        super().__init__(require_nonlinear_pk=require_nonlinear_pk)
        self.p_ml = p_ml
        self.p_mnl = p_mnl
        self.dist = dist

    def set_params_from_numcosmo(
        self, mset: Ncm.MSet
    ):  # pylint: disable-msg=too-many-locals
        """Return a PyCCLCosmologyConstants object with parameters equivalent to
        those read from NumCosmo."""

        hi_cosmo = mset.peek(Nc.HICosmo.id())

        if self.p_ml is not None:
            self.p_ml.prepare_if_needed(hi_cosmo)
        if self.p_mnl is not None:
            self.p_mnl.prepare_if_needed(hi_cosmo)
        self.dist.prepare_if_needed(hi_cosmo)

        h = hi_cosmo.h()  # pylint: disable-msg=C0103
        Omega_b = hi_cosmo.Omega_b0()  # pylint: disable-msg=invalid-name
        Omega_c = hi_cosmo.Omega_c0()  # pylint: disable-msg=invalid-name
        Omega_k = hi_cosmo.Omega_k0()  # pylint: disable-msg=invalid-name
        Neff = hi_cosmo.Neff()  # pylint: disable-msg=invalid-name
        T_gamma0 = hi_cosmo.T_gamma0()  # pylint: disable-msg=invalid-name

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
            w0 = hi_cosmo.props.w  # pylint: disable-msg=C0103
            wa = 0.0  # pylint: disable-msg=C0103
        elif isinstance(hi_cosmo, Nc.HICosmoDECpl):
            w0 = hi_cosmo.props.w0  # pylint: disable-msg=C0103
            wa = hi_cosmo.props.w1  # pylint: disable-msg=C0103
        else:
            raise ValueError(f"NumCosmo object {type(hi_cosmo)} not supported.")

        hiprim = hi_cosmo.peek_submodel_by_mid(Nc.HIPrim.id())
        if not hiprim:
            raise ValueError("NumCosmo object must include a HIPrim object.")
        if not isinstance(hiprim, Nc.HIPrimPowerLaw):
            raise ValueError(
                f"NumCosmo HIPrim object type {type(hiprim)} not supported."
            )

        A_s = hiprim.SA_Ampl()  # pylint: disable-msg=invalid-name
        n_s = hiprim.props.n_SA  # pylint: disable-msg=invalid-name

        # pylint: disable=duplicate-code
        self.set_params(
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

        if self.p_ml:
            p_m_spline = self.p_ml.get_spline_2d(hi_cosmo)
            z = np.array(p_m_spline.xv.dup_array())  # pylint: disable-msg=invalid-name
            k = np.array(p_m_spline.yv.dup_array())

            scale = self.redshift_to_scale_factor(z)
            p_k = np.transpose(
                np.array(p_m_spline.zm.dup_array()).reshape(len(k), len(z))
            )
            p_k = self.redshift_to_scale_factor_p_k(p_k)

            ccl_args["pk_linear"] = {
                "a": scale,
                "k": k,
                "delta_matter:delta_matter": p_k,
            }

        if self.p_mnl:
            p_mnl_spline = self.p_mnl.get_spline_2d(hi_cosmo)
            # pylint: disable-next=invalid-name
            z = np.array(p_mnl_spline.xv.dup_array())
            k = np.array(p_mnl_spline.yv.dup_array())

            scale_mpnl = self.redshift_to_scale_factor(z)
            p_mnl = np.transpose(
                np.array(p_mnl_spline.zm.dup_array()).reshape(len(k), len(z))
            )
            p_mnl = self.redshift_to_scale_factor_p_k(p_mnl)

            ccl_args["pk_nonlin"] = {
                "a": scale_mpnl,
                "k": k,
                "delta_matter:delta_matter": p_mnl,
            }
        elif self.require_nonlinear_pk:
            ccl_args["nonlinear_model"] = "halofit"
        else:
            ccl_args["nonlinear_model"] = None

        d_spline = self.dist.comoving_distance_spline.peek_spline()
        z_dist = np.array(d_spline.get_xv().dup_array())
        c_dist = np.array(d_spline.get_yv().dup_array())

        chi = np.flip(c_dist) * hi_cosmo.RH_Mpc()
        scale_distances = self.redshift_to_scale_factor(z_dist)
        h_over_h0 = np.array([hi_cosmo.E(z) for z in reversed(z_dist)])

        ccl_args["background"] = build_ccl_background_dict(
            a=scale_distances, chi=chi, h_over_h0=h_over_h0
        )
        ccl_args["background"] = {
            "a": scale_distances,
            "chi": chi,
            "h_over_h0": h_over_h0,
        }

        return ccl_args


class NumCosmoData(Ncm.Data):
    """NumCosmoData is a subclass of Ncm.Data and implements NumCosmo likelihood
    object virtual methods using the prefix :python:`do_`. This class implement
    a general likelihood."""

    def __init__(
        self,
        likelihood: Likelihood,
        tools: ModelingTools,
        mapping: MappingNumCosmo,
        model_list: List[str],
    ):
        super().__init__()
        self.likelihood: Likelihood = likelihood
        self.tools: ModelingTools = tools
        self.dof: int = 100
        self.len: int = 100
        self.mapping = mapping
        self.model_list: List[str] = model_list
        self.ccl_cosmo: Optional[ccl.Cosmology] = None

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
        # Implements the virtual Ncm.Data method `begin`.
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
        firecrown_params = ParamsMap()
        self.likelihood.reset()
        self.tools.reset()

        self.mapping.set_params_from_numcosmo(mset)
        ccl_args = self.mapping.calculate_ccl_args(mset)
        self.ccl_cosmo = ccl.CosmologyCalculator(**self.mapping.asdict(), **ccl_args)

        for model_ns in self.model_list:
            mid = mset.get_id_by_ns(model_ns)
            model = mset.peek(mid)
            param_names = model.param_names()
            model_dict = {
                param: model.param_get_by_name(param) for param in param_names
            }
            shared_keys = set(model_dict).intersection(firecrown_params)
            if len(shared_keys) > 0:
                raise RuntimeError(
                    f"The following keys `{shared_keys}` appear "
                    f"in more than one model used by the "
                    f"module {self.model_list}."
                )
            firecrown_params = ParamsMap({**firecrown_params, **model_dict})

        self.likelihood.update(firecrown_params)
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

    def __init__(
        self,
        likelihood: ConstGaussian,
        tools: ModelingTools,
        mapping: MappingNumCosmo,
        model_list: List[str],
    ):
        """Initialize a NumCosmoGaussCov object representing a Gaussian likelihood
        with a constant covariance."""
        cov = likelihood.get_cov()
        nrows, ncols = cov.shape
        assert nrows == ncols

        super().__init__(n_points=nrows)

        self.likelihood: ConstGaussian = likelihood
        self.tools: ModelingTools = tools
        self.mapping = mapping
        self.model_list: List[str] = model_list
        self.ccl_cosmo: Optional[ccl.Cosmology] = None

        self.dof = nrows
        self.len = nrows
        self.peek_cov().set_from_array(cov.flatten())  # pylint: disable-msg=no-member

        data_vector = likelihood.get_data_vector()
        assert len(data_vector) == ncols
        self.y.set_array(data_vector)  # pylint: disable-msg=no-member

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
        firecrown_params = ParamsMap()
        self.likelihood.reset()
        self.tools.reset()

        self.mapping.set_params_from_numcosmo(mset)
        ccl_args = self.mapping.calculate_ccl_args(mset)
        self.ccl_cosmo = ccl.CosmologyCalculator(**self.mapping.asdict(), **ccl_args)

        for model_ns in self.model_list:
            mid = mset.get_id_by_ns(model_ns)
            model = mset.peek(mid)
            param_names = model.param_names()
            model_dict = {
                param: model.param_get_by_name(param) for param in param_names
            }
            shared_keys = set(model_dict).intersection(firecrown_params)
            if len(shared_keys) > 0:
                raise RuntimeError(
                    f"The following keys `{shared_keys}` appear "
                    f"in more than one model used by the "
                    f"module {self.model_list}."
                )
            firecrown_params = ParamsMap({**firecrown_params, **model_dict})

        self.likelihood.update(firecrown_params)
        self.tools.prepare(self.ccl_cosmo)

    # pylint: disable-next=invalid-name,arguments-differ
    def do_mean_func(self, _, mean_vector):
        """
        Implements the virtual `Ncm.DataGaussCov` method `mean_func`.
        This method should compute the theoretical mean for the gaussian
        distribution.
        """

        theory_vector = self.likelihood.compute_theory_vector(self.tools)
        mean_vector.set_array(theory_vector)


class NumCosmoFactory:
    """NumCosmo likelihood class. This class provide the necessary factory methods
    to create NumCosmo+firecrown likelihoods."""

    def __init__(
        self,
        likelihood_source: str,
        build_parameters: NamedParameters,
        model_list: List[str],
        mapping: MappingNumCosmo,
    ):
        likelihood, tools = load_likelihood(likelihood_source, build_parameters)

        self.mapping: MappingNumCosmo = mapping
        if isinstance(likelihood, ConstGaussian):
            self.data: Ncm.Data = NumCosmoGaussCov(
                likelihood, tools, mapping, model_list
            )
        else:
            self.data = NumCosmoData(likelihood, tools, mapping, model_list)
        self.data.set_init(True)

    def get_data(self) -> Ncm.Data:
        """This method return the appropriated Ncm.Data class to be used by NumCosmo."""
        return self.data

    def get_mapping(self) -> MappingNumCosmo:
        """This method return the current MappingNumCosmo."""
        return self.mapping
