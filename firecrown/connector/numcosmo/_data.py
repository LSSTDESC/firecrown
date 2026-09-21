"""Support of Firecrown likelihood use in NumCosmo.

The subpackages and modules in this package depend upon NumCosmo, and can not
be used without an installation of NumCosmo.
"""

from typing import Protocol

from numcosmo_py import GObject, Ncm

from firecrown.connector.numcosmo._mapping import MappingNumCosmo
from firecrown.connector.numcosmo._mapping import (
    _named_parameters_to_var_dict,
    _var_dict_to_named_parameters,
)
from firecrown.likelihood import (
    ConstGaussian,
    Likelihood,
    NamedParameters,
    load_likelihood,
)
from firecrown.modeling_tools import (
    CCLCreationMode,
    ModelingTools,
)
from firecrown.updatable import UpdatableUsageRecord, handle_unused_params
from firecrown.connector.numcosmo._parameters import create_params_map


# NumCosmoData and NumCosmoGaussCov below share identical model_list/nc_mapping
# GObject.Property getters and setters, but they do not share a common base
# class that declares _model_list/_nc_mapping, so neither class can be used as
# the concrete `self` type. These Protocols express the actual structural
# requirement ("any object with this one attribute") shared by both classes.
#
# They are also not a formality: GObject.Property's stub types `getter`/
# `setter` as Callable[[Any], Any], so it accepts a function with an entirely
# unannotated `self` just as readily. But an unannotated `self` makes the
# whole function untyped, and this project's mypy configuration enables
# check_untyped_defs without disallow_untyped_defs, so mypy checks such a
# function's body treating `self` as Any -- silently permitting a typo like
# `self._modle_list` with no diagnostic. Annotating `self` against these
# Protocols keeps the getter/setter bodies checked against a real attribute
# contract instead of silently degrading to Any.
class _HasModelList(Protocol):
    """Structural type for NumCosmo data classes with a model name list."""

    _model_list: list[str]


class _HasNcMapping(Protocol):
    """Structural type for NumCosmo data classes with a MappingNumCosmo."""

    _nc_mapping: MappingNumCosmo | None


def _get_model_list(self: _HasModelList) -> list[str]:
    """Return the list of model names.

    Shared GObject.Property getter for :class:`NumCosmoData` and
    :class:`NumCosmoGaussCov`.

    :returns: the names of the contained models
    """
    return self._model_list  # pylint: disable=protected-access


def _set_model_list(self: _HasModelList, value: list[str]) -> None:
    """Set the list of model names.

    Shared GObject.Property setter for :class:`NumCosmoData` and
    :class:`NumCosmoGaussCov`.

    :param value: the new list of model names
    """
    self._model_list = value  # pylint: disable=protected-access


def _get_nc_mapping(self: _HasNcMapping) -> MappingNumCosmo | None:
    """Return the MappingNumCosmo object.

    Shared GObject.Property getter for :class:`NumCosmoData` and
    :class:`NumCosmoGaussCov`.

    :returns: the MappingNumCosmo object, or None
    """
    return self._nc_mapping  # pylint: disable=protected-access


def _set_nc_mapping(self: _HasNcMapping, value: MappingNumCosmo | None) -> None:
    """Set the MappingNumCosmo object.

    Shared GObject.Property setter for :class:`NumCosmoData` and
    :class:`NumCosmoGaussCov`.

    :param value: the new value for the MappingNumCosmo object
    """
    self._nc_mapping = value  # pylint: disable=protected-access


class NumCosmoData(Ncm.Data):
    """NumCosmoData is a subclass of Ncm.Data.

    This subclass also implements NumCosmo likelihood object virtual methods using
    the prefix `do_`. This class implements a general likelihood.
    """

    __gtype_name__ = "FirecrownNumCosmoData"

    def __init__(self) -> None:
        """Initialize a NumCosmoData object.

        Default values are provided for all attributes; most default are None.
        """
        super().__init__()
        self.likelihood: Likelihood
        self.tools: ModelingTools
        self._model_list: list[str]
        self._nc_mapping: MappingNumCosmo | None
        self._likelihood_source: None | str = None
        self._likelihood_build_parameters: None | NamedParameters = None
        self._starting_deserialization: bool = False
        self.dof: int = 100
        self.len: int = 100
        self.set_init(True)

    # getter/setter are shared with NumCosmoGaussCov below; see the
    # _HasModelList/_HasNcMapping Protocols above for why they are typed
    # against a Protocol instead of either concrete class.
    model_list = GObject.Property(
        # GObject.TYPE_STRV is a raw GType constant with no Python type
        # equivalent; the Property stub's `type` parameter cannot express it.
        type=GObject.TYPE_STRV,  # type: ignore[arg-type]
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_model_list,
        setter=_set_model_list,
    )

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

        :returns: the filename of the likelihood factory function
        """
        return self._likelihood_source

    def _set_likelihood_source(self, value: None | str) -> None:
        """Set the likelihood string defining the factory function.

        :param value: the filename of the likelihood factory function
        """
        assert value is not None
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
        return _named_parameters_to_var_dict(self._likelihood_build_parameters)

    def _set_likelihood_build_parameters(self, value: None | Ncm.VarDict) -> None:
        """Set the likelihood build parameters.

        :param value: the parameters used to build the likelihood
        """
        self._likelihood_build_parameters = _var_dict_to_named_parameters(value)

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
        nc_mapping: MappingNumCosmo | None,
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

        :returns: the number of data points in the likelihood
        """
        return self.len

    def do_get_dof(self) -> int:  # pylint: disable-msg=arguments-differ
        """Implements the virtual Ncm.Data method get_dof.

        :returns: the number of degrees of freedom in the likelihood
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

        updated_records: list[UpdatableUsageRecord] = []
        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            assert self._nc_mapping is not None
            self._nc_mapping.set_params_from_numcosmo(
                mset, self.tools.ccl_factory.amplitude_parameter
            )
            params_map = create_params_map(
                self.model_list, mset, self._nc_mapping.mapping
            )
            self.likelihood.update(params_map, updated_records)
            self.tools.update(params_map, updated_records)
            self.tools.prepare(
                calculator_args=self._nc_mapping.calculate_ccl_args(mset)
            )
        else:
            params_map = create_params_map(self.model_list, mset, None)
            self.likelihood.update(params_map, updated_records)
            self.tools.update(params_map, updated_records)
            self.tools.prepare()

        handle_unused_params(
            params=params_map,
            updated_records=updated_records,
            raise_on_unused=self.likelihood.raise_on_unused_parameter,
        )

    def do_m2lnL_val(  # pylint: disable-msg=arguments-differ
        self, _: Ncm.MSet
    ) -> float:
        """Implements the virtual method `m2lnL`.

        This method should calculate the value of the likelihood for
        the model set `mset`.

        :param _: unused, but required by interface
        """
        loglike = self.likelihood.compute_loglike_for_sampling(self.tools)
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
        self._nc_mapping: MappingNumCosmo | None
        self._likelihood_source: None | str = None
        self._likelihood_build_parameters: None | NamedParameters = None
        self._starting_deserialization: bool = False

    # getter/setter are shared with NumCosmoData above; see the
    # _HasModelList/_HasNcMapping Protocols above for why they are typed
    # against a Protocol instead of either concrete class.
    model_list = GObject.Property(
        # GObject.TYPE_STRV is a raw GType constant with no Python type
        # equivalent; the Property stub's `type` parameter cannot express it.
        type=GObject.TYPE_STRV,  # type: ignore[arg-type]
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_model_list,
        setter=_set_model_list,
    )

    nc_mapping = GObject.Property(
        type=MappingNumCosmo,
        flags=GObject.ParamFlags.READWRITE | GObject.ParamFlags.CONSTRUCT,
        getter=_get_nc_mapping,
        setter=_set_nc_mapping,
    )

    def _will_calculate_power_spectra(self) -> bool:
        """Return whether the likelihood will calculate power spectra.

        :returns: whether the likelihood will calculate power spectra
        """
        if self._nc_mapping is None:
            return False
        return (self._nc_mapping.p_ml is not None) or (
            self._nc_mapping.p_mnl is not None
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

        if (
            (self.tools.ccl_factory.creation_mode != CCLCreationMode.DEFAULT)
            and self._will_calculate_power_spectra()
            and (not self.tools.ccl_factory.allow_multiple_camb_instances)
        ):
            raise RuntimeError(
                "If Firecrown is using CCL to calculate the cosmology, then "
                "NumCosmo should not be configured to calculate power spectra."
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

        :returns: the filename of the likelihood factory function
        """
        return self._likelihood_source

    def _set_likelihood_source(self, value: None | str) -> None:
        """Set the likelihood string defining the factory function.

        The value should not be None. The type declaration is required
        because of the nature of the C interface being wrapped, but it
        is a programming error to pass a null pointer.

        :param value: the filename of the likelihood factory function
        """
        self._likelihood_source = value
        if value is None:
            return
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

        :returns: the likelihood build parameters
        """
        return _named_parameters_to_var_dict(self._likelihood_build_parameters)

    def _set_likelihood_build_parameters(self, value: None | Ncm.VarDict) -> None:
        """Set the likelihood build parameters.

        :param value: the likelihood build parameters
        """
        self._likelihood_build_parameters = _var_dict_to_named_parameters(value)
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
        nc_mapping: MappingNumCosmo | None,
        likelihood_source: None | str = None,
        likelihood_build_parameters: None | NamedParameters = None,
    ) -> "NumCosmoGaussCov":
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

        :returns: the number of data points in the likelihood
        """
        return self.len

    def do_get_dof(self) -> int:  # pylint: disable-msg=arguments-differ
        """Implements the virtual `Ncm.Data` method `get_dof`.

        :returns: the number of degrees of freedom in the likelihood
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

        if self.tools.ccl_factory.creation_mode == CCLCreationMode.DEFAULT:
            assert self._nc_mapping is not None
            self._nc_mapping.set_params_from_numcosmo(
                mset, self.tools.ccl_factory.amplitude_parameter
            )
            params_map = create_params_map(
                self._model_list, mset, self._nc_mapping.mapping
            )
            self.likelihood.update(params_map)
            self.tools.update(params_map)
            self.tools.prepare(
                calculator_args=self._nc_mapping.calculate_ccl_args(mset)
            )
        else:
            assert self._nc_mapping is None
            params_map = create_params_map(self._model_list, mset, None)
            self.likelihood.update(params_map)
            self.tools.update(params_map)
            self.tools.prepare()

    # pylint: disable-next=arguments-differ
    def do_mean_func(self, _: Ncm.MSet, vp: Ncm.Vector) -> None:
        """Implements the virtual `Ncm.DataGaussCov` method `mean_func`.

        This method should compute the theoretical mean for the gaussian
        distribution.

        :param _: unused, but required by interface
        :param vp: the vector to set
        """
        theory_vector = self.likelihood.compute_theory_vector(self.tools)
        vp.set_array(theory_vector)
