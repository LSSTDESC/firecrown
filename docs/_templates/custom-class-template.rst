{{ fullname | escape | underline}}

.. role:: python(code)
   :language: python

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :special-members: __all__, __add__, __eq__
   :private-members:

   .. autoclasstoc::
