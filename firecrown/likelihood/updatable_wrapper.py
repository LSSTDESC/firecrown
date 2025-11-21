from firecrown import parameters
from firecrown.updatable import Updatable, UpdatableCollection


class UpdatableParameters(Updatable):
    """
    Class to store and pass updatable parameters to cluster objects.

    Attributes
    ----------
    updatable_parameters: list
        Name of updatable parameters.
    """

    def __init__(self, updatable_parameters):
        """
        Parameters
        ----------
        updatable_parameters: list
            Name of updatable parameters.
        """
        super().__init__()
        self.updatable_parameters = updatable_parameters

    def init_parameters(self, cluster_object):
        """
        Instanciate all parameters (uses parameters.register_new_updatable_parameter)

        Parameters
        ----------
        cluster_object: object
            cluster object to get the defalt parameters from.
        """
        for par_name in self.updatable_parameters:
            setattr(
                self,
                par_name,
                parameters.register_new_updatable_parameter(
                    default_value=getattr(cluster_object, par_name)
                ),
            )

    def export_parameters(self, cluster_object):
        """
        Passes internal parameters to cluster object.

        Parameters
        ----------
        cluster_object: list
            cluster object to export internal parameters to.
        """
        for par_name in self.updatable_parameters:
            setattr(
                cluster_object,
                par_name,
                getattr(self, par_name),
            )


class UpdatableClusterObjects(Updatable):
    """
    Class to store and pass updatable parameters of all to cluster objects
    in a cluster recipe.

    Attributes
    ----------
    cluster_objects_configs: tuple
        List of dictionaries containing configuration on which parameters
        of each cluster objects in the recipe that will be updated.


    Examples
    --------

    cluster_objects_configs = (
        {
            "attribute_name": "mass_distribution",
            "parameters": ["mu_p0", "mu_p1", "mu_p2", "sigma_p0", "sigma_p1", "sigma_p2"],
        },
        {
            "attribute_name": "cluster_theory",
            "parameters": [],
            "has_cosmo": True,
        },
        {
            "attribute_name": "completeness",
            "parameters": ["ac_nc", "bc_nc", "ac_rc", "bc_rc"],
        },
        {
            "attribute_name": "purity",
            "parameters": ["ap_nc", "bp_nc", "ap_rc", "bp_rc"],
        },
    )
    """

    def __init__(self, cluster_objects_configs):
        """
        Parameters
        ----------
        cluster_objects_configs: tuple
            List of dictionaries containing configuration on which parameters
            of each cluster objects in the recipe that will be updated. Each
            dictionary should contain the keys:

            - attribute_name: name of the attribute in the recipe.
            - parameters: list name of parameters that should be updatable.
            - has_cosmo (optional, defalut=False): if this attribute has an internal cosmology.
        """
        super().__init__()
        self.cluster_objects_configs = cluster_objects_configs
        self.my_updatables = UpdatableCollection()
        for conf in self.cluster_objects_configs:
            setattr(
                self,
                conf["attribute_name"],
                UpdatableParameters(conf["parameters"]),
            )
            self.my_updatables.append(getattr(self, conf["attribute_name"]))

    def init_all_parameters(self, cluster_recipe):
        """
        Instanciate all parameters (uses parameters.register_new_updatable_parameter)

        Parameters
        ----------
        cluster_recipe: recipe object
            Recipe containing all cluster objects (as attributes) to get the defalt parameters from.
        """
        for conf in self.cluster_objects_configs:
            getattr(self, conf["attribute_name"]).init_parameters(
                getattr(cluster_recipe, conf["attribute_name"])
            )

    def export_all_parameters(self, cluster_recipe, cosmo):
        """
        Passes internal parameters to cluster object.

        Parameters
        ----------
        cluster_object: list
            Recipe containing all cluster objects (as attributes) to export internal parameters to.
        """
        for conf in self.cluster_objects_configs:
            _recipe_attribute = getattr(cluster_recipe, conf["attribute_name"])
            getattr(self, conf["attribute_name"]).export_parameters(_recipe_attribute)
            if conf.get("has_cosmo", False):
                _recipe_attribute.cosmo = cosmo
