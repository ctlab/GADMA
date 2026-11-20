"""
Fixtures that are common for all tests.
"""
import copy

import pytest

import gadma
from gadma.cli import settings as default_settings
from gadma.engines import engine as engine_module
from gadma.utils import variables

# Options of engines that are kept in their classes
ENGINE_OPTIONS = ["extrapolation", "kwargs", "ld_kwargs", "n_processes",
                  "supported_models", "supported_data"]


def get_global_values():
    """
    Returns values that GADMA keeps in classes and modules, so they are
    common for the whole process: default domains and random generators of
    variables, options of engines, flags of available packages and default
    values of settings. Settings change some of them (e.g. `min_t` or
    `number_of_processes`).

    :returns: list of (owner, name of attribute, value).
    """
    values = []
    for cls in vars(variables).values():
        if isinstance(cls, type) and issubclass(cls, variables.Variable):
            for name in ["default_domain", "default_rand_gen"]:
                if name in vars(cls):
                    values.append((cls, name, getattr(cls, name)))
    engine_classes = []
    for engine_cls in engine_module._registered_engines.values():
        for cls in engine_cls.__mro__:
            if cls not in engine_classes:
                engine_classes.append(cls)
    for cls in engine_classes:
        for name in ENGINE_OPTIONS:
            if name in vars(cls):
                values.append((cls, name, getattr(cls, name)))
    for name, value in vars(gadma).items():
        if name.endswith("_available"):
            values.append((gadma, name, value))
    for name, value in vars(default_settings).items():
        if not name.startswith("_"):
            values.append((default_settings, name, value))
    return values


@pytest.fixture(autouse=True)
def restore_global_values():
    """
    Returns values that are common for the whole process (see
    :func:`get_global_values`) back after every test. Otherwise a test that
    changes them, e.g. bounds of variables, changes them for all tests that
    are run after it.
    """
    saved = [(owner, name, copy.deepcopy(value))
             for owner, name, value in get_global_values()]
    yield
    for owner, name, value in saved:
        setattr(owner, name, value)
