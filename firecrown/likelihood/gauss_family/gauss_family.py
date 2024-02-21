"""

Gaussian Family Module
======================

Some notes.

"""

from __future__ import annotations

from enum import Enum
from functools import wraps
from typing import Optional, Sequence, Callable, Union, TypeVar
from typing import final
import warnings

from typing_extensions import ParamSpec
import numpy as np
import numpy.typing as npt
import scipy.linalg

import sacc

from ...parameters import ParamsMap
from ..likelihood import Likelihood
from ...modeling_tools import ModelingTools
from ...updatable import UpdatableCollection
from .statistic.statistic import Statistic, GuardedStatistic
from ...utils import save_to_sacc


class State(Enum):
    """The states used in GaussFamily."""

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
    initial: Union[State, list[State]],
    terminal: Optional[State] = None,
    failure_message: str,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """This decorator wraps a method, and enforces state machine behavior. If
    the object is not in one of the states in initial, an
    AssertionError is raised with the given failure_message.
    If terminal is None the state of the object is not modified.
    If terminal is not None and the call to the wrapped method returns
    normally the state of the object is set to terminal.
    """
    initials: list[State]
    if isinstance(initial, list):
        initials = initial
    else:
        initials = [initial]

    def decorator_enforce_states(func: Callable[P, T]) -> Callable[P, T]:
        """This part of the decorator is the closure that actually contains the
        values of initials, terminal, and failure_message.
        """

        @wraps(func)
        def wrapper_repeat(*args: P.args, **kwargs: P.kwargs) -> T:
            """This part of the decorator is the actual wrapped method. It is
            responsible for confirming a correct initial state, and
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
    """GaussFamily is an abstract class. It is the base class for all likelihoods
    based on a chi-squared calculation. It provides an implementation of
    Likelihood.compute_chisq. Derived classes must implement the abstract method
    compute_loglike, which is inherited from Likelihood.

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
    ):
        """Initialize the base class parts of a GaussFamily object."""
        super().__init__()
        self.state: State = State.INITIALIZED
        if len(statistics) == 0:
            raise ValueError("GaussFamily requires at least one statistic")

        for i, s in enumerate(statistics):
            if not isinstance(s, Statistic):
                raise ValueError(
                    f"statistics[{i}] is not an instance of Statistic: {s}"
                    f"it is a {type(s)} instead."
                )

        self.statistics: UpdatableCollection[GuardedStatistic] = UpdatableCollection(
            GuardedStatistic(s) for s in statistics
        )
        self.cov: Optional[npt.NDArray[np.float64]] = None
        self.cholesky: Optional[npt.NDArray[np.float64]] = None
        self.inv_cov: Optional[npt.NDArray[np.float64]] = None
        self.cov_index_map: Optional[dict[int, int]] = None
        self.theory_vector: Optional[npt.NDArray[np.double]] = None
        self.data_vector: Optional[npt.NDArray[np.double]] = None

    @enforce_states(
        initial=State.READY,
        terminal=State.UPDATED,
        failure_message="read() must be called before update()",
    )
    def _update(self, _: ParamsMap) -> None:
        """Handle the state resetting required by :class:`GaussFamily`
        likelihoods. Any derived class that needs to implement :meth:`_update`
        for its own reasons must be sure to do what this does: check the state
        at the start of the method, and change the state at the end of the
        method."""

    @enforce_states(
        initial=[State.UPDATED, State.COMPUTED],
        terminal=State.READY,
        failure_message="update() must be called before reset()",
    )
    def _reset(self) -> None:
        """Handle the state resetting required by :class:`GaussFamily`
        likelihoods. Any derived class that needs to implement :meth:`reset`
        for its own reasons must be sure to do what this does: check the state
        at the start of the method, and change the state at the end of the
        method."""
        self.theory_vector = None

    @enforce_states(
        initial=State.INITIALIZED,
        terminal=State.READY,
        failure_message="read() must only be called once",
    )
    def read(self, sacc_data: sacc.Sacc) -> None:
        """Read the covariance matrix for this likelihood from the SACC file."""

        if sacc_data.covariance is None:
            msg = (
                f"The {type(self).__name__} likelihood requires a covariance, "
                f"but the SACC data object being read does not have one."
            )
            raise RuntimeError(msg)

        covariance = sacc_data.covariance.dense

        indices_list = []
        data_vector_list = []
        for stat in self.statistics:
            stat.read(sacc_data)
            if stat.statistic.sacc_indices is None:
                raise RuntimeError(
                    f"The statistic {stat.statistic} has no sacc_indices."
                )
            indices_list.append(stat.statistic.sacc_indices.copy())
            data_vector_list.append(stat.statistic.get_data_vector())

        indices = np.concatenate(indices_list)
        data_vector = np.concatenate(data_vector_list)
        cov = np.zeros((len(indices), len(indices)))

        for new_i, old_i in enumerate(indices):
            for new_j, old_j in enumerate(indices):
                cov[new_i, new_j] = covariance[old_i, old_j]

        self.data_vector = data_vector
        self.cov_index_map = {old_i: new_i for new_i, old_i in enumerate(indices)}
        self.cov = cov
        self.cholesky = scipy.linalg.cholesky(self.cov, lower=True)
        self.inv_cov = np.linalg.inv(cov)

    @enforce_states(
        initial=[State.READY, State.UPDATED, State.COMPUTED],
        failure_message="read() must be called before get_cov()",
    )
    @final
    def get_cov(
        self, statistic: Union[Statistic, list[Statistic], None] = None
    ) -> npt.NDArray[np.float64]:
        """Gets the current covariance matrix.

        :param statistic: The statistic for which the sub-covariance matrix
        should be return. If not specified, return the covariance of all
        statistics.
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
            temp = [self.cov_index_map[idx] for idx in stat.sacc_indices]
            indices += temp
        ixgrid = np.ix_(indices, indices)

        return self.cov[ixgrid]

    @final
    @enforce_states(
        initial=[State.READY, State.UPDATED, State.COMPUTED],
        failure_message="read() must be called before get_data_vector()",
    )
    def get_data_vector(self) -> npt.NDArray[np.float64]:
        """Get the data vector from all statistics and concatenate in the right
        order."""
        assert self.data_vector is not None
        return self.data_vector

    @final
    @enforce_states(
        initial=State.UPDATED,
        terminal=State.COMPUTED,
        failure_message="update() must be called before compute_theory_vector()",
    )
    def compute_theory_vector(self, tools: ModelingTools) -> npt.NDArray[np.float64]:
        """Computes the theory vector using the current instance of pyccl.Cosmology.

        :param tools: Current ModelingTools object
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
        """Get the theory vector from all statistics and concatenate in the right
        order."""
        assert (
            self.theory_vector is not None
        ), "theory_vector is None after compute_theory_vector() has been called"
        return self.theory_vector

    @final
    @enforce_states(
        initial=State.UPDATED,
        failure_message="update() must be called before compute()",
    )
    def compute(
        self, tools: ModelingTools
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculate and return both the data and theory vectors."""
        warnings.simplefilter("always", DeprecationWarning)
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
        """Calculate and return the chi-squared for the given cosmology."""
        theory_vector: npt.NDArray[np.float64]
        data_vector: npt.NDArray[np.float64]
        residuals: npt.NDArray[np.float64]
        try:
            theory_vector = self.compute_theory_vector(tools)
            data_vector = self.get_data_vector()
        except NotImplementedError:
            data_vector, theory_vector = self.compute(tools)

        assert len(data_vector) == len(theory_vector)
        residuals = data_vector - theory_vector

        x = scipy.linalg.solve_triangular(self.cholesky, residuals, lower=True)
        chisq = np.dot(x, x)
        return chisq

    @enforce_states(
        initial=[State.READY, State.UPDATED, State.COMPUTED],
        failure_message="read() must be called before get_sacc_indices()",
    )
    def get_sacc_indices(
        self, statistic: Union[Statistic, list[Statistic], None] = None
    ) -> npt.NDArray[np.int64]:
        """Get the SACC indices of the statistic or list of statistics. If no
        statistic is given, get the indices of all statistics of the likelihood."""
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
