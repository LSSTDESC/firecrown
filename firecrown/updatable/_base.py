"""Base class for updatable objects."""

from __future__ import annotations
from abc import ABC
from collections.abc import Iterable
from typing import Any, final

from typing_extensions import assert_never

from firecrown.parameters import (
    InternalParameter,
    ParamsMap,
    RequiredParameters,
    SamplerParameter,
)

from firecrown.parameters import DerivedParameterCollection, UpdatableUsageRecord

from firecrown.updatable._exceptions import MissingSamplerParameterError
from firecrown.updatable._types import UpdatableProtocol


class Updatable(ABC):
    """Abstract class Updatable is the base class for Updatable objects in Firecrown.

    Any class in Firecrown that supports updating from a ParamsMap should
    inherit from :class:`Updatable`. Such classes are expected to change state
    only in through their implementation of _update (including any other private
    methods used to implement _update). Other functions should not change the
    data of Updatable objects.
    """

    def __init__(self, parameter_prefix: None | str = None) -> None:
        """Updatable initialization.

        Parameters created by firecrown.parameters.create will have a prefix
        that is given by the prefix argument to the Updatable constructor. This
        prefix is used to create the full name of the parameter. If `parameter_prefix`
        is None, then the parameter will have no prefix.

        :param parameter_prefix: prefix for all parameters in this Updatable
        """
        self._updated: bool = False
        self._returned_derived: bool = False
        self._sampler_parameters: list[SamplerParameter] = []
        self._internal_parameters: dict[str, InternalParameter] = {}
        self._updatables: list[UpdatableProtocol] = []
        self.parameter_prefix: None | str = parameter_prefix

    def __setattr__(self, key: str, value: Any) -> None:
        """Set the attribute named :attr:`key` to the supplied `value`.

        There is special handling for two types: :class:`SamplerParameter`
        and :class:`InternalParameter`.

        We also keep track of all :class:`Updatable` instance variables added,
        appending a reference to each to :attr:`self._updatables` as well as
        storing the attribute directly.

        :param key: name of the attribute
        :param value: value for the attribute
        """
        if isinstance(value, UpdatableProtocol):
            self._updatables.append(value)
        elif isinstance(value, Iterable):
            # Consider making this a recursive call that handles nested
            # iterables.
            for v in value:
                if isinstance(v, UpdatableProtocol):
                    self._updatables.append(v)

        if isinstance(value, (InternalParameter, SamplerParameter)):
            self.set_parameter(key, value)
        else:
            super().__setattr__(key, value)

    def set_parameter(
        self, key: str, value: InternalParameter | SamplerParameter
    ) -> None:
        """Sets the parameter to the given value.

        Assure this InternalParameter or SamplerParameter has not already
        been set, and then set it.

        :param key: name of the attribute
        :param value: value for the attribute
        """
        match value:
            case SamplerParameter():
                value.set_fullname(self.parameter_prefix, key)
                self.set_sampler_parameter(value)
            case InternalParameter():
                self.set_internal_parameter(key, value)
            case _ as unreachable:
                assert_never(unreachable)

    def set_internal_parameter(self, key: str, value: InternalParameter) -> None:
        """Assure this InternalParameter has not already been set, and then set it.

        :param key: name of the attribute
        :param value: value for the attribute
        """
        if not isinstance(value, InternalParameter):
            raise TypeError(
                "Can only add InternalParameter objects to internal_parameters"
            )
        if key in self._internal_parameters or hasattr(self, key):
            raise ValueError(
                f"attribute {key} already set in {self} "
                f"from a parameter supplied in the likelihood factory code"
            )
        self._internal_parameters[key] = value
        super().__setattr__(key, value.get_value())

    def set_sampler_parameter(self, value: SamplerParameter) -> None:
        """Assure this SamplerParameter has not already been set, and then set it.

        :param value: value for the attribute
        """
        if not isinstance(value, SamplerParameter):
            raise TypeError(
                "Can only add SamplerParameter objects to sampler_parameters"
            )

        if value in self._sampler_parameters or hasattr(self, value.name):
            raise ValueError(
                f"attribute {value.name} already set in {self} "
                f"from a parameter read from the sampler"
            )
        self._sampler_parameters.append(value)
        super().__setattr__(value.name, None)

    @final
    def update(
        self,
        params: ParamsMap,
        updated_record: list[UpdatableUsageRecord] | None = None,
    ) -> None:
        """Update self by calling to prepare for the next MCMC sample.

        We first update the values of sampler parameters from the values in
        `params`. An error will be raised if any of self's sampler
        parameters can not be found in `params` or if any internal
        parameters are provided in `params`.

        We then use the `params` to update each contained :class:`Updatable` or
        :class:`UpdatableCollection` object. The method :meth:`_update` is
        called to give subclasses an opportunity to do any other preparation
        for the next MCMC sample.

        :param params: new parameter values
        """
        if self._updated:
            if updated_record is not None:
                updated_record.append(
                    UpdatableUsageRecord(
                        cls=type(self).__name__,
                        prefix=self.parameter_prefix,
                        obj_id=id(self),
                        sampler_params=[],
                        internal_params=[],
                        child_records=[],
                        already_updated=True,
                    )
                )
            return

        internal_params = self._internal_parameters.keys() & params.keys()

        if internal_params:
            raise TypeError(
                f"Items of type InternalParameter cannot be modified through "
                f"update, but {','.join(internal_params)} was specified."
            )

        for parameter in self._sampler_parameters:
            try:
                value = params.get_from_full_name(parameter.fullname)
            except KeyError as exc:
                raise MissingSamplerParameterError(parameter.fullname) from exc
            setattr(self, parameter.name, value)

        child_records: list[UpdatableUsageRecord] | None = None
        if updated_record is not None:
            child_records = []
            updated_record.append(
                UpdatableUsageRecord(
                    cls=type(self).__name__,
                    prefix=self.parameter_prefix,
                    obj_id=id(self),
                    sampler_params=[sp.name for sp in self._sampler_parameters],
                    internal_params=list(self._internal_parameters.keys()),
                    child_records=child_records,
                )
            )

        # When `updated_record` is not None, `child_records` is created as an empty
        # list and passed to the `UpdatableUsageRecord` above. This means both the
        # local variable `child_records` and the corresponding field in the record
        # refer to the *same list object*. As each child `update` call appends to
        # `updated_record=child_records`, those new entries automatically appear inside
        # the parent record's `child_records` field. This works because lists are
        # passed by reference (object identity), not by value.
        for item in self._updatables:
            item.update(params, updated_record=child_records)

        self._update(params)
        # We mark self as updated only after all the internal updates have
        # worked.
        self._updated = True

    def is_updated(self) -> bool:
        """Determine if the object has been updated.

        A default-constructed Updatable has not been updated. After `update`,
        but before `reset`, has been called the object is updated. After
        `reset` has been called, the object is not currently updated.

        :return:  True if the object is currently updated, and False if not.
        """
        return self._updated

    @final
    def reset(self) -> None:
        """Reset the updatable.

        Clean up self by clearing the _updated status and resetting all
        internals. We call the abstract method _reset to allow derived classes
        to clean up any additional internals.

        Each MCMC framework connector should call this after handling an MCMC
        sample.
        """
        # If we have not been updated, there is nothing to do.
        if not self._updated:
            return

        # We reset in the inverse order, first the contained updatables, then
        # the current object.
        for item in self._updatables:
            item.reset()

        # Reset the sampler parameters to None.
        for parameter in self._sampler_parameters:
            setattr(self, parameter.name, None)

        self._updated = False
        self._returned_derived = False
        self._reset()

    def _update(self, params: ParamsMap) -> None:
        """Method for auxiliary updates to be made to an updatable.

        Do any updating other than calling :meth:`update` on contained
        :class:`Updatable` objects. Implement this method in a subclass only when it
        has something to do. If the supplied :class:`ParamsMap` is lacking a required
        parameter, an implementation should raise a `TypeError`.

        This default implementation does nothing.

        :param params: a new set of parameter values
        """

    def _reset(self) -> None:
        """Abstract method implemented by all concrete classes to update self.

        Concrete classes must override this, resetting themselves.

        The base class implementation does nothing.
        """

    @final
    def required_parameters(self) -> RequiredParameters:
        """Returns all information about parameters required by this object.

        This object returned contains the information for all parameters
        defined in the implementing class, and any additional parameters.

        :return: a RequiredParameters object containing all relevant parameters
        """
        sampler_parameters = RequiredParameters(self._sampler_parameters)
        additional_parameters = self._required_parameters()

        for item in self._updatables:
            additional_parameters = additional_parameters + item.required_parameters()

        return sampler_parameters + additional_parameters

    @final
    def get_params_names(self) -> list[str]:
        """Return the names of the parameters required by this object.

        The order of the returned names is arbitrary.

        :return: a list of parameter names
        """
        return list(self.required_parameters().get_params_names())

    def _required_parameters(self) -> RequiredParameters:
        """Return a RequiredParameters object containing the information for this class.

        This method can be overridden by subclasses to add
        additional parameters. The default implementation returns an empty
        RequiredParameters object. This is only implemented to allow

        The base class implementation returns a list with all SamplerParameter
        objects properties.

        :return: a RequiredParameters containing all relevant parameters
        """
        return RequiredParameters([])

    @final
    def get_derived_parameters(
        self,
    ) -> None | DerivedParameterCollection:
        """Returns a collection of derived parameters.

        This occurs once per iteration of the statistical analysis. First call returns
        the DerivedParameterCollection, further calls return None.

        :return: a collection of derived parameters, or None
        """
        if not self._updated:
            raise RuntimeError(
                "Derived parameters can only be obtained after update has been called."
            )

        if self._returned_derived:
            return None

        self._returned_derived = True
        derived_parameters = self._get_derived_parameters()

        for item in self._updatables:
            derived_parameters = derived_parameters + item.get_derived_parameters()

        return derived_parameters

    def _get_derived_parameters(self) -> DerivedParameterCollection:
        """Returns the derived parameters of an implementation.

        Derived classes can override this, returning a DerivedParameterCollection
        containing the derived parameters for the class. The default implementation
        returns an empty DerivedParameterCollection.

        :return: a collection of derived parameters
        """
        return DerivedParameterCollection([])
