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

    def __init__(self, recipe_attribute_name, updatable_parameters):
        """
        Parameters
        ----------
        updatable_parameters: list
            Name of updatable parameters.
        """
        super().__init__()
        self.recipe_attribute_name = recipe_attribute_name
        self.updatable_parameters = updatable_parameters

    def _ini_file_par_name(self, par_name):
        return f"{self.recipe_attribute_name}.{par_name}"

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
                self._ini_file_par_name(par_name),
                parameters.register_new_updatable_parameter(
                    default_value=cluster_object.parameters[par_name]
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
            cluster_object.parameters[par_name] = getattr(
                self, self._ini_file_par_name(par_name)
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
            "recipe_attribute_name": "mass_distribution",
            "parameters": ["mu0", "mu1", "mu2", "sigma0", "sigma1", "sigma2"],
        },
        {
            "recipe_attribute_name": "cluster_theory",
            "parameters": ["cluster_concentration"], # if wl profile
            "has_cosmo": True,
        },
        {
            "recipe_attribute_name": "completeness",
            "parameters": ["a_n", "b_n", "a_logm_piv", "b_logm_piv"],
        },
        {
            "recipe_attribute_name": "purity",
            "parameters": ["a_n", "b_n", "a_logm_piv", "b_logm_piv"],
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

            - recipe_attribute_name: name of the attribute in the recipe.
            - parameters: list name of parameters that should be updatable.
            - has_cosmo (optional, defalut=False): if this attribute has an internal cosmology.

            BinnedCluster has a function that automatically creates the correct
            list according to the elements in the recipe.
        """
        super().__init__()
        self.cluster_objects_configs = cluster_objects_configs
        self.my_updatables = UpdatableCollection()
        for conf in self.cluster_objects_configs:
            setattr(
                self,
                conf["recipe_attribute_name"],
                UpdatableParameters(conf["recipe_attribute_name"], conf["parameters"]),
            )
            self.my_updatables.append(getattr(self, conf["recipe_attribute_name"]))

    def init_all_parameters(self, cluster_recipe):
        """
        Instanciate all parameters (uses parameters.register_new_updatable_parameter)

        Parameters
        ----------
        cluster_recipe: recipe object
            Recipe containing all cluster objects (as attributes) to get the defalt parameters from.
        """
        for conf in self.cluster_objects_configs:
            getattr(self, conf["recipe_attribute_name"]).init_parameters(
                getattr(cluster_recipe, conf["recipe_attribute_name"])
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
            _recipe_attribute = getattr(cluster_recipe, conf["recipe_attribute_name"])
            getattr(self, conf["recipe_attribute_name"]).export_parameters(
                _recipe_attribute
            )
            if conf.get("has_cosmo", False):
                _recipe_attribute.cosmo = cosmo
