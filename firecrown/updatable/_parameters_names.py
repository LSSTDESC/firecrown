"""Utilities for parameter names and validation."""


def parameter_get_full_name(prefix: None | str, param: str) -> str:
    """Form a full parameter name from the given (optional) prefix and name.

    Parameter names, as stored in SACC, for example, contain an optional
    prefix; if a prefix is present, it will be separated from the name by
    an underscore.

    Prefixes and names should avoid containing embedded underscores. This
    is currently not enforced in the code.

    The parameter name can not be empty, even if accompanied by a prefix;
    this is enforced in the code.

    :param prefix: optional prefix
    :param param: name
    :return: full name
    :raises ValueError: if the parameter name is empty
    """
    if len(param) == 0:
        raise ValueError("param must not be an empty string")

    if prefix:
        return f"{prefix}_{param}"
    return param


def _validate_params_map_value(name: str, value: float | list[float]) -> None:
    """Check if the value is a float or a list of floats.

    Raises a TypeError if the value is not a float or a list of floats.

    :param name: name of the parameter
    :param value: value to be checked
    """
    if not isinstance(value, (float, list)):
        raise TypeError(
            f"Value for parameter {name} is not a float or a list of floats: "
            f"{type(value)}"
        )

    if isinstance(value, list):
        if not all(isinstance(v, float) for v in value):
            raise TypeError(
                f"Value for parameter {name} is not a float or a list of floats: "
                f"{type(value)}"
            )
