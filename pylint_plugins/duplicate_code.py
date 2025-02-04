"""Firecrown-specific plugin to ignore duplicate code warnings in recipes."""

from astroid import MANAGER, scoped_nodes
from pylint.lint import PyLinter


def register(_: PyLinter):
    """The PyLint plugin registration function.

    This one doesn't have to do anything."""


def transform(mod):
    """Transform the module's code.

    If the module name begins with `firecrown.models.cluster.recipes.`, add
    a `# pylint: disable=duplicate-code` comment to the top of the module.
    """
    if "firecrown.models.cluster.recipes." not in mod.name:
        return

    c = mod.stream().read()
    c = b"# pylint: disable=duplicate-code\n" + c

    # pylint will read from `.file_bytes` attribute later when tokenization
    mod.file_bytes = c


MANAGER.register_transform(scoped_nodes.Module, transform)
