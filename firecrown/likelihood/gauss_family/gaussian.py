"""Provides GaussFamily concrete types.

"""

from __future__ import annotations
import numpy as np
import sacc

from .gauss_family import GaussFamily, State
from ...modeling_tools import ModelingTools


class ConstGaussian(GaussFamily):
    """A Gaussian log-likelihood with a constant covariance matrix."""

    def compute_loglike(self, tools: ModelingTools):
        """Compute the log-likelihood."""

        return -0.5 * self.compute_chisq(tools)

    def make_realization(
        self, sacc_data: sacc.Sacc, add_noise: bool = True, strict: bool = True
    ) -> sacc.Sacc:
        assert (
            self.state == State.UPDATED
        ), "update() must be called before make_realization()"

        if not self.computed_theory_vector:
            raise RuntimeError(
                "The theory vector has not been computed yet. "
                "Call compute_theory_vector first."
            )

        new_sacc = sacc_data.copy()

        sacc_indices_list = []
        for stat in self.statistics:
            assert stat.statistic.sacc_indices is not None
            sacc_indices_list.append(stat.statistic.sacc_indices.copy())

        sacc_indices = np.concatenate(sacc_indices_list)
        theory_vector = self.get_theory_vector()
        assert len(sacc_indices) == len(theory_vector)

        if strict:
            if set(sacc_indices.tolist()) != set(sacc_data.indices()):
                raise RuntimeError(
                    "The predicted data does not cover all the data in the "
                    "sacc object. To write only the calculated predictions, "
                    "set strict=False."
                )

        # Adding Gaussian noise defined by the covariance matrix.
        if add_noise:
            assert self.cholesky is not None
            new_data_vector = theory_vector + np.dot(
                self.cholesky, np.random.randn(len(theory_vector))
            )
        else:
            new_data_vector = theory_vector

        for prediction_idx, sacc_idx in enumerate(sacc_indices):
            new_sacc.data[sacc_idx].value = new_data_vector[prediction_idx]

        return new_sacc
