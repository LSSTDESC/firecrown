"""NumCosmo parameter-map construction API."""

from numcosmo_py import Ncm

from firecrown.connector.mapping import Mapping
from firecrown.updatable import ParamsMap


def create_params_map(
    model_list: list[str], mset: Ncm.MSet, mapping: Mapping | None
) -> ParamsMap:
    """Create a ParamsMap from a NumCosmo MSet.

    All the models named in model_list must be in the model set `mset`, or a
    RuntimeError will be raised.

    :param model_list: list of model names
    :param mset: the NumCosmo MSet object from which to get the parameters
    :returns: a ParamsMap containing the parameters of the models in model_list
    """
    params_map = ParamsMap()
    for model_ns in model_list:
        mid = mset.get_id_by_ns(model_ns)
        if mid < 0:
            raise RuntimeError(f"Model name {model_ns} was not found in the model set.")
        model = mset.peek(mid)
        # Since we have already verified that the model name exists in the
        # model set, if the model is not found we have encountered an
        # unrecoverable error.
        assert model is not None

        param_names = model.param_names()
        model_dict = {param: model.param_get_by_name(param) for param in param_names}
        params_map.update(model_dict)

    if mapping is not None:
        params_map.update(mapping.asdict())

    return params_map
