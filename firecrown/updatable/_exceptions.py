"""Exceptions for the updatable module."""

from __future__ import annotations


class MissingSamplerParameterError(RuntimeError):
    """Error for when a required parameter is missing.

    Raised when an Updatable fails to be updated because the ParamsMap supplied for the
    update is missing a parameter that should have been provided by the sampler.
    """

    def __init__(self, parameter: str) -> None:
        """Create the error, with a meaningful error message.

        :param parameter: name of the missing parameter
        """
        self.parameter = parameter
        msg = (
            f"The parameter `{parameter}` is required to update "
            f"something in this likelihood.\nIt should have been supplied "
            f"by the sampling framework.\nThe object being updated was:\n"
            f"{self}\n"
        )
        super().__init__(msg)
