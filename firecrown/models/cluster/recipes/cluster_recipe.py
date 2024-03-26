"""Module for defining the ClusterRecipe class."""

from abc import ABC, abstractmethod
from typing import Optional

from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.binning import SaccBin
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.updatable import Updatable, UpdatableCollection


class ClusterRecipe(Updatable, ABC):
    """Abstract class defining a cluster recipe.

    A cluster recipe is a combination of different cluster theoretrical predictions
    and models that produces a single prediction for an observable.
    """

    def __init__(self, parameter_prefix: Optional[str] = None) -> None:
        super().__init__(parameter_prefix)
        self.my_updatables: UpdatableCollection = UpdatableCollection()

    @abstractmethod
    def evaluate_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        this_bin: SaccBin,
        sky_area: float,
        average_on: Optional[ClusterProperty] = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe."""
