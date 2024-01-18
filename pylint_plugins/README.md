### Custom Pylint Plugins
Pylint does not have any built in functionality to selectively apply rules to subdirectories.  

As an example, if we want to ignore a pylint warning in only one class/directory/module, we would need to create a custom `pylintrc` file and manually exclude this file from the parent pylint runs.

Custom plugins allow us to selectively ignore warnings. This directory holds the custom plugins we load into pylint through the `pylintrc` files. 

### Table of Contents

* `duplicate_code`: A custom plugin that will ignore the `duplicate-code` warning.  Currently the only classes we allow to ignore this message is `firecrown.models.cluster.recipes`.