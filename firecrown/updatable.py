from __future__ import annotations
from typing import List, final
from abc import ABC, abstractmethod
from .parameters import ParamsMap, RequiredParameters


class Updatable(ABC):
    @final
    def update(self, params: ParamsMap):
        self._update(params)

    @abstractmethod
    def _update(self, params: ParamsMap):
        pass

    @abstractmethod
    def required_parameters(self) -> RequiredParameters:
        pass


class UpdatableCollection(List):
    @final
    def update(self, params: ParamsMap):
        for updatable in self:
            updatable.update(params)

    @final
    def required_parameters(self) -> RequiredParameters:

        rp = RequiredParameters([])
        for updatable in self:
            rp = rp + updatable.required_parameters()

        return rp
