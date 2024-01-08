from abc import ABC, abstractmethod
from typing import Optional

from firecrown.models.cluster.abundance import ClusterAbundance
from firecrown.models.cluster.binning import SaccBin
from firecrown.models.cluster.properties import ClusterProperty
from firecrown.updatable import Updatable, UpdatableCollection


class ClusterRecipe(Updatable, ABC):
    def __init__(self, parameter_prefix: str | None = None) -> None:
        super().__init__(parameter_prefix)
        self.my_updatables: UpdatableCollection = UpdatableCollection()

    @abstractmethod
    def evaluate_theory_prediction(
        self,
        cluster_theory: ClusterAbundance,
        bin: SaccBin,
        sky_area: float,
        average_on: Optional[ClusterProperty] = None,
    ) -> float:
        pass
