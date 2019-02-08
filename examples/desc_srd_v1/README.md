# DESC SRD v1 Models

To run this example, first type

```bash
$ pip install -e .
```

from this directory.

Then you can type

```bash
$ python generate_srd_data.py
$ firecrown compute srd_v1_gen.yaml
Watch out! Here comes a firecrown!
analysis id: <analysis id>
loglike: None
$ python move_gen_data.py <analysis id>
```

to build the example data.

Finally, type

```bash
$ firecrown compute srd_v1_model.yaml
```

to compute the model.

You can make plots of the data compared to the SRD using the notebook.
