from astroid import MANAGER
from astroid import scoped_nodes
from pylint.lint import PyLinter


def register(linter: PyLinter):
    pass


def transform(mod):
    if "firecrown.models.cluster.recipes." not in mod.name:
        return
    print(mod.name)
    c = mod.stream().read()
    c = b"# pylint: disable=duplicate-code\n" + c

    # pylint will read from `.file_bytes` attribute later when tokenization
    mod.file_bytes = c


MANAGER.register_transform(scoped_nodes.Module, transform)
