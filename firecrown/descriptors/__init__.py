"""Type validation as used in connectors.

Validators are created using the constructor for each class.
Access to the data done through the object name, not through any named function.
Setting the data is validated with the class's `validate` function; the user does
not need to call any special functions.

Validators are intended for use in class definitions. An example is a class that
has an attribute `x` that is required to be a float in the range
[1.0, 3.0], but is optional and has a default value of None:

.. code:: python

    class SampleValidatedThing:
        x = TypeFloat(1.0, 3.0, allow_none=True)

        def __init__(self):
            self.x = None

"""

from firecrown.descriptors._float import TypeFloat
from firecrown.descriptors._string import TypeString

__all__ = [
    "TypeFloat",
    "TypeString",
]
