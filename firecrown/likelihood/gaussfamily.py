"""Support for the family of Gaussian likelihoods."""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Sequence, Callable, TypeVar
from typing import final
import warnings

from typing_extensions import ParamSpec
import numpy as np
import numpy.typing as npt
import scipy.linalg

import sacc

from firecrown.parameters import ParamsMap
from firecrown.likelihood.likelihood import Likelihood
from firecrown.modeling_tools import ModelingTools
from firecrown.updatable import UpdatableCollection
from firecrown.likelihood.statistic import (
    Statistic,
    GuardedStatistic,
)
from firecrown.utils import save_to_sacc


class State(Enum):
    """The states used in GaussFamily.

    GaussFamily and all subclasses enforce a statemachine behavior based on
    these states to ensure that the necessary initialization and setup is done
    in the correct order.
    """

    INITIALIZED = 1
    READY = 2
    UPDATED = 3
    COMPUTED = 4


T = TypeVar("T")
P = ParamSpec("P")


# See https://peps.python.org/pep-0612/ and
# https://stackoverflow.com/questions/66408662/type-annotations-for-decorators
# for how to specify the types of *args and **kwargs, and the return type of
# the method being decorated.


# Beware
def enforce_states(
    *,
    initial: State | list[State],
    terminal: None | State = None,
    failure_message: str,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """This decorator wraps a method, and enforces state machine behavior.

    If the object is not in one of the states in initial, an
    AssertionError is raised with the given failure_message.
    If terminal is None the state of the object is not modified.
    If terminal is not None and the call to the wrapped method returns
    normally the state of the object is set to terminal.

    :param initial: The initial states allowable for the wrapped method
    :param terminal: The terminal state ensured for the wrapped method. None
        indicates no state change happens.
    :param failure_message: The failure message for the AssertionError raised
    :return: The wrapped method
    """
    initials: list[State]
    if isinstance(initial, list):
        initials = initial
    else:
        initials = [initial]

    def decorator_enforce_states(func: Callable[P, T]) -> Callable[P, T]:
        """Part of the decorator which is the closure.

        This closure is what actually contains the values of initials, terminal, and
        failure_message.

        :param func: The method to be wrapped
        :return: The wrapped method
        """

        @wraps(func)
        def wrapper_repeat(*args: P.args, **kwargs: P.kwargs) -> T:
            """Part of the decorator which is the actual wrapped method.

            It is responsible for confirming a correct initial state, and
            establishing the correct final state if the wrapped method
            succeeds.
            """
            # The odd use of args[0] instead of self seems to be the only way
            # to have both the Python runtime and mypy agree on what is being
            # passed to the method, and to allow access to the attribute
            # 'state'. Recall that the syntax:
            #       o.foo()
            # calls a bound function object accessible as o.foo; this bound
            # function object calls the function foo() passing 'o' as the
            # first argument, self.
            assert isinstance(args[0], GaussFamily)
            assert args[0].state in initials, failure_message
            value = func(*args, **kwargs)
            if terminal is not None:
                args[0].state = terminal
            return value

        return wrapper_repeat

    return decorator_enforce_states


class GaussFamily(Likelihood):
    """GaussFamily is the base class for likelihoods based on a chi-squared calculation.

    It provides an implementation of Likelihood.compute_chisq. Derived classes must
    implement the abstract method compute_loglike, which is inherited from Likelihood.

    GaussFamily (and all classes that inherit from it) must abide by the the
    following rules regarding the order of calling of methods.

      1. after a new object is created, :meth:`read` must be called before any
         other method in the interfaqce.
      2. after :meth:`read` has been called it is legal to call
         :meth:`get_data_vector`, or to call :meth:`update`.
      3. after :meth:`update` is called it is then legal to call
         :meth:`calculate_loglike` or :meth:`get_data_vector`, or to reset
         the object (returning to the pre-update state) by calling
         :meth:`reset`. It is also legal to call :meth:`compute_theory_vector`.
      4. after :meth:`compute_theory_vector` is called it is legal to call
         :meth:`get_theory_vector` to retrieve the already-calculated theory
         vector.

    This state machine behavior is enforced through the use of the decorator
    :meth:`enforce_states`, above.
    """

    def __init__(
        self,
        statistics: Sequence[Statistic],
    ) -> None:
        """Initialize the base class parts of a GaussFamily object.

        :param statistics: A list of statistics to be include in chisquared calculations
        """
        super().__init__()
        self.state: State = State.INITIALIZED
        if len(statistics) == 0:
            raise ValueError("GaussFamily requires at least one statistic")

        for i, s in enumerate(statistics):
            if not isinstance(s, Statistic):
                raise ValueError(
                    f"statistics[{i}] is not an instance of Statistic."
                    f" It is a {type(s)}."
                )

        self.statistics: UpdatableCollection[GuardedStatistic] = UpdatableCollection(
            GuardedStatistic(s) for s in statistics
        )
        self.cov: None | npt.NDArray[np.float64] = None
        self.cholesky: None | npt.NDArray[np.float64] = None
        self.inv_cov: None | npt.NDArray[np.float64] = None
        self.cov_index_map: None | dict[int, int] = None
        self.theory_vector: None | npt.NDArray[np.double] = None
        self.data_vector: None | npt.NDArray[np.double] = None

    @classmethod
    def create_ready(
        cls, statistics: Sequence[Statistic], covariance: npt.NDArray[np.float64]
    ) -> GaussFamily:
        """Create a GaussFamily object in the READY state.

        :param statistics: A list of statistics to be include in chisquared calculations
        :param covariance: The covariance matrix of the statistics
        :return: A ready GaussFamily object
        """
        obj = cls(statistics)
        obj._set_covariance(covariance)
        obj.state = State.READY
        return obj

    @enforce_states(
        initial=State.READY,
        terminal=State.UPDATED,
        failure_message="read() must be called before update()",
    )
    def _update(self, _: ParamsMap) -> None:
        """Handle the state resetting required by :class:`GaussFamily` likelihoods.

        Any derived class that needs to implement :meth:`_update`
        for its own reasons must be sure to do what this does: check the state
        at the start of the method, and change the state at the end of the
        method.

        :param _: a ParamsMap object, not used
        """

    @enforce_states(
        initial=[State.UPDATED, State.COMPUTED],
        terminal=State.READY,
        failure_message="update() must be called before reset()",
    )
    def _reset(self) -> None:
        """Handle the state resetting required by :class:`GaussFamily` likelihoods.

        Any derived class that needs to implement :meth:`reset`
        for its own reasons must be sure to do what this does: check the state
        at the start of the method, and change the state at the end of the
        method.
        """
        self.theory_vector = None

    @enforce_states(
        initial=State.INITIALIZED,
        terminal=State.READY,
        failure_message="read() must only be called once",
    )
    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file.

        :param sacc_data: The SACC data object to be read
        """
        if sacc_data.covariance is None:
            msg = (
                f"The {type(self).__name__} likelihood requires a covariance, "
                f"but the SACC data object being read does not have one."
            )
            raise RuntimeError(msg)

        for stat in self.statistics:
            stat.read(sacc_data)

        covariance = sacc_data.covariance.dense

        self._set_covariance(covariance)

    def _set_covariance(self, covariance: npt.NDArray[np.float64]) -> None:
        """Set the covariance matrix.

        This method is used to set the covariance matrix and perform the
        necessary calculations to prepare the likelihood for computation.

        :param covariance: The covariance matrix for this likelihood
        """
        indices_list = []
        data_vector_list = []
        for stat in self.statistics:
            if not stat.statistic.ready:
                raise RuntimeError(
                    f"The statistic {stat.statistic} is not ready to be used."
                )
            if stat.statistic.sacc_indices is None:
                raise RuntimeError(
                    f"The statistic {stat.statistic} has no sacc_indices."
                )
            indices_list.append(stat.statistic.sacc_indices.copy())
            data_vector_list.append(stat.statistic.get_data_vector())

        indices = np.concatenate(indices_list).astype(int)
        data_vector = np.concatenate(data_vector_list)
        cov = np.zeros((len(indices), len(indices)))

        largest_index = int(np.max(indices))

        if not (
            covariance.ndim == 2
            and covariance.shape[0] == covariance.shape[1]
            and largest_index < covariance.shape[0]
        ):
            raise ValueError(
                f"The covariance matrix has shape {covariance.shape}, "
                f"but the expected shape is at least "
                f"{(largest_index + 1, largest_index + 1)}."
            )

        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                cov[new_i, new_j] = covariance[old_i, old_j]

        self.data_vector = data_vector
        self.cov_index_map = {old_i: new_i for new_i, old_i in enumerate(indices)}
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True).astype(np.float64)
        self.inv_cov = np.linalg.inv(cov).astype(np.float64)

    @final
    @enforce_states(
        initial=[State.READY, State.UPDATED, State.COMPUTED],
        failure_message="read() must be called before get_cov()",
    )
    def get_cov(
        self, statistic: Statistic | list[Statistic] | None = None
    ) -> npt.NDArray[np.float64]:
        """Gets the current covariance matrix.

        :param statistic: The statistic for which the sub-covariance matrix
            should be returned. If not specified, return the covariance of all
            statistics.
        :return: The covariance matrix (or portion thereof)
        """
        assert self.cov is not None
        if statistic is None:
            return self.cov

        assert self.cov_index_map is not None
        if isinstance(statistic, Statistic):
            statistic_list = [statistic]
        else:
            statistic_list = statistic
        indices: list[int] = []
        for stat in statistic_list:
            assert stat.sacc_indices is not None
            temp = [self.cov_index_map[int(idx)] for idx in stat.sacc_indices]
            indices += temp
        ixgrid = np.ix_(indices, indices)

        return self.cov[ixgrid]

    @final
    @enforce_states(
        initial=[State.READY, State.UPDATED, State.COMPUTED],
        failure_message="read() must be called before get_data_vector()",
    )
    def get_data_vector(self) -> npt.NDArray[np.float64]:
        """Get the data vector from all statistics in the right order.

        :return: The data vector
        """
        assert self.data_vector is not None
        return self.data_vector.astype(np.float64)

    @final
    @enforce_states(
        initial=[State.UPDATED, State.COMPUTED],
        terminal=State.COMPUTED,
        failure_message="update() must be called before compute_theory_vector()",
    )
    def compute_theory_vector(self, tools: ModelingTools) -> npt.NDArray[np.float64]:
        """Computes the theory vector using the current instance of pyccl.Cosmology.

        :param tools: Current ModelingTools object
        :return: The computed theory vector
        """
        theory_vector_list: list[npt.NDArray[np.float64]] = [
            stat.compute_theory_vector(tools) for stat in self.statistics
        ]
        self.theory_vector = np.concatenate(theory_vector_list)
        return self.theory_vector

    @final
    @enforce_states(
        initial=State.COMPUTED,
        failure_message="compute_theory_vector() must be called before "
        "get_theory_vector()",
    )
    def get_theory_vector(self) -> npt.NDArray[np.float64]:
        """Get the already-computed theory vector from all statistics.

        :return: The theory vector, with all statistics in the right order
        """
        assert (
            self.theory_vector is not None
        ), "theory_vector is None after compute_theory_vector() has been called"
        return self.theory_vector.astype(np.float64)

    @final
    @enforce_states(
        initial=State.UPDATED,
        failure_message="update() must be called before compute()",
    )
    def compute(
        self, tools: ModelingTools
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate and return both the data and theory vectors.

        This method is dprecated and will be removed in a future version of Firecrown.

        :param tools: the ModelingTools to be used in the calculation of the
            theory vector
        :return: a tuple containing the data vector and the theory vector
        """
        warnings.warn(
            "The use of the `compute` method on Statistic is deprecated."
            "The Statistic objects should implement `get_data` and "
            "`compute_theory_vector` instead.",
            category=DeprecationWarning,
        )
        return self.get_data_vector(), self.compute_theory_vector(tools)

    @final
    @enforce_states(
        initial=[State.UPDATED, State.COMPUTED],
        terminal=State.COMPUTED,
        failure_message="update() must be called before compute_chisq()",
    )
    def compute_chisq(self, tools: ModelingTools) -> float:
        """Calculate and return the chi-squared for the given cosmology.

        :param tools: the ModelingTools to be used in the calculation of the
            theory vector
        :return: the chi-squared
        """
        theory_vector: npt.NDArray[np.float64]
        data_vector: npt.NDArray[np.float64]
        residuals: npt.NDArray[np.float64]
        try:
            theory_vector = self.compute_theory_vector(tools)
            data_vector = self.get_data_vector()
        except NotImplementedError:
            data_vector, theory_vector = self.compute(tools)

        assert len(data_vector) == len(theory_vector)
        residuals = np.array(data_vector - theory_vector, dtype=np.float64)

        assert self.cholesky is not None
        x = scipy.linalg.solve_triangular(self.cholesky, residuals, lower=True)
        chisq = np.dot(x, x)
        return chisq

    @enforce_states(
        initial=[State.READY, State.UPDATED, State.COMPUTED],
        failure_message="read() must be called before get_sacc_indices()",
    )
    def get_sacc_indices(
        self, statistic: Statistic | list[Statistic] | None = None
    ) -> npt.NDArray[np.int64]:
        """Get the SACC indices of the statistic or list of statistics.

        If no statistic is given, get the indices of all statistics of the likelihood.

        :param statistics: The statistic or list of statistics for which the
            SACC indices are desired
        :return: The SACC indices
        """
        if statistic is None:
            statistic = [stat.statistic for stat in self.statistics]
        if isinstance(statistic, Statistic):
            statistic = [statistic]

        sacc_indices_list = []
        for stat in statistic:
            assert stat.sacc_indices is not None
            sacc_indices_list.append(stat.sacc_indices.copy())

        sacc_indices = np.concatenate(sacc_indices_list)

        return sacc_indices

    @enforce_states(
        initial=State.COMPUTED,
        failure_message="compute_theory_vector() must be called before "
        "make_realization()",
    )
    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True, strict: bool = True
    ) -> sacc.Sacc:
        """Create a new realization of the model.

        :param sacc_data: The SACC data object containing the covariance matrix
            to be read
        :param add_noise: If True, add noise to the realization.
        :param strict: If True, check that the indices of the realization cover
            all the indices of the SACC data object.
        :return: The SACC data object containing the new realization
        """
        sacc_indices = self.get_sacc_indices()

        if add_noise:
            new_data_vector = self.make_realization_vector()
        else:
            new_data_vector = self.get_theory_vector()

        new_sacc = save_to_sacc(
            sacc_data=sacc_data,
            data_vector=new_data_vector,
            indices=sacc_indices,
            strict=strict,
        )

        return new_sacc
