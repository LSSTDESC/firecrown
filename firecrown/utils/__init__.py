import importlib
import pathlib


def load_module_by_path(filepath):
    # Determine the base name of the file
    # no directory or
    p = pathlib.Path(filepath)
    name = p.stem

    # From the 3.6 docs, via
    # https://stackoverflow.com/questions/27381264/python-3-4-how-to-import-a-module-given-the-full-path
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod
