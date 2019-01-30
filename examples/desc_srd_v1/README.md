# DESC SRC v1 Models

To run this example, first type

```bash
$ pip install -e .
```

from this directory.

Then you can type

```bash
$ generate_srd_data.py
$ firecrown compute srd_v1_gen.yaml
$ python move_gen_data.py
```

to build the example data.

Finally, type

```bash
$ firecrown compute srd_v1_model.yaml
```

to compute the model.
