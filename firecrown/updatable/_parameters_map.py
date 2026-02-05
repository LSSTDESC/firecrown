"""Parameter map functionality and unused parameter handling."""

import copy
import warnings

from ._parameters_names import _validate_params_map_value, parameter_get_full_name


class ParamsMap:
    """A dict-like object in which all keys are strings and values are floats.

    The recommended access method is get_from_prefix_param, rather than indexing
    with square brackets like x[].
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ParamsMap.

        :param args: arguments
        :param kwargs: keyword arguments
        """
        self.params: dict[str, float] = dict(*args, **kwargs)
        for name, value in self.params.items():
            _validate_params_map_value(name, value)

        self.lower_case: bool = False
        self.used_keys: set[str] = set()

    def __getitem__(self, key: str) -> float:
        """Return the value for the given key.

        If the key has not been used, add it to the set of used keys.

        :param key: key
        :return: value
        """
        if key in self.params:
            self.used_keys.add(key)
        return self.params[key]

    def __setitem__(self, key: str, value: float) -> None:
        """Set the value for the given key.

        :param key: key
        :param value: value
        """
        self.params[key] = value

    def __contains__(self, key: str) -> bool:
        """Return True if the key is in the map, False otherwise.

        :param key: key
        :return: True if the key is in the map, False otherwise
        """
        return key in self.params

    def copy(self) -> "ParamsMap":
        """Return a shallow copy of the ParamsMap."""
        result = ParamsMap()
        result.params = self.params.copy()
        result.used_keys = self.used_keys.copy()
        result.lower_case = self.lower_case
        return result

    def items(self):
        """Return an iterator over the items in the dictionary.

        :return: an iterator over the items in the dictionary
        """
        return self.params.items()

    def union(self, other: "ParamsMap") -> "ParamsMap":
        """Return a new ParamsMap that is the union of self and other.

        If the same key is used in both self and other, the values in
        both self and other must be equal.

        :param other: other ParamsMap
        :return: a new ParamsMap that is the union of self and other
        """
        assert isinstance(other, self.__class__)
        my_keys = set(self.params.keys())
        other_keys = set(other.params.keys())
        for common_key in my_keys.intersection(other_keys):
            if self.params[common_key] != other.params[common_key]:
                raise ValueError(
                    f"Key {common_key} has different values in self and other."
                )
        result = copy.deepcopy(self)
        result.params.update(other.params)
        result.used_keys.update(other.used_keys)
        return result

    def update(self, d: dict[str, float]) -> None:
        """Update self with the values from d.

        This will raise an error if any of the keys in d are already in self.

        :param d: dictionary
        """
        for key, value in d.items():
            if key in self.params:
                raise ValueError(f"Key {key} is already present in the ParamsMap.")
            self.params[key] = value

    def use_lower_case_keys(self, enable: bool) -> None:
        """Control whether keys will be translated into lower case.

        If `enable` is True, such translation will be done.
        This can help make sure code works with CosmoSIS, because such translation
        is done inside CosmoSIS itself.

        :param enable: whether to enable or disable this option
        """
        self.lower_case = enable

    def get(self, key: str, default: float | None = None) -> float:
        """Return the value for the given key, or default if the key is not found.

        :param key: key
        :param default: default value, used if the key is not found
        :return: value
        """
        if key in self.params:
            self.used_keys.add(key)
        if default is None:
            return self.params[key]
        return self.params.get(key, default)

    def get_from_full_name(self, full_name: str) -> float:
        """Return the parameter identified by the full name.

        Raises a KeyError if the parameter is not found.
        """
        if full_name in self.params.keys():
            self.used_keys.add(full_name)
            return self.params[full_name]

        if self.lower_case:
            full_name_lower = full_name.lower()
            if full_name_lower in self.params.keys():
                self.used_keys.add(full_name_lower)
                return self.params[full_name_lower]

        raise KeyError(f"Key {full_name} not found.")

    def get_from_prefix_param(self, prefix: None | str, param: str) -> float:
        """Return the parameter identified by the optional prefix and parameter name.

        See parameter_get_full_name for rules on the forming of prefix and name.
        Raises a KeyError if the parameter is not found.
        """
        fullname = parameter_get_full_name(prefix, param)
        return self.get_from_full_name(fullname)

    def get_unused_keys(self) -> set[str]:
        """Return the set of keys that have not been used.

        This is the set of keys that are not in self.used_keys.
        """
        return set(self.params.keys()) - self.used_keys

    def keys(self) -> set[str]:
        """Return the set of keys in the map.

        :return: set of keys
        """
        return set(self.params.keys())


def handle_unused_params(
    params: ParamsMap,
    updated_records: list,
    raise_on_unused: bool = False,
):
    """Check for unused keys in the parameters map."""
    unused_keys = params.get_unused_keys()
    if unused_keys:
        message = (
            f"Unused keys in parameters: {sorted(unused_keys)}.\n"
            "This may indicate a missing statistic or systematic.\n"
            "You may also have a modeling mismatch, e.g. you have specified "
            "sampling of neutrino masses but did not configure CAMB to use "
            "massive neutrinos.\n"
        )
        # Add log lines from updated records
        log_lines = []
        for record in updated_records:
            log_lines += record.get_log_lines()
        message += "\n".join(log_lines)

        if raise_on_unused:
            raise ValueError(message)

        warnings.warn(message)
