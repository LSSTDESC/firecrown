"""
Provides a trivial likelihood factory function for testing purposes.
This module should be loaded by the test_load_likelihood_submodule test.
It should raise an exception because the factory function does not define
a build_likelihood as a Callable.
"""


build_likelihood = "I am not a function"
